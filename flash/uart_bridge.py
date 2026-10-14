"""Serialize UART requests on the Docker host and release the port for flashing."""
import json
import math
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class UARTError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


class UARTBridge:
    def __init__(self, port, baud=115200, serial_factory=None):
        if serial_factory is None:
            import serial
            serial_factory = serial.Serial
        self.port, self.baud, self.serial_factory = port, baud, serial_factory
        self.lock = threading.RLock()
        self.serial = None
        self.board = None
        self.flashing = False
        self.rx = b''

    def _close(self):
        serial, self.serial = self.serial, None
        self.board = None
        self.rx = b''
        if serial is not None:
            serial.close()

    def status(self):
        with self.lock:
            return dict(connected=self.serial is not None, flashing=self.flashing,
                        port=self.port, baud=self.baud, board=self.board)

    def disconnect(self):
        with self.lock:
            self._close()
            return self.status()

    @contextmanager
    def flash_session(self):
        with self.lock:
            if self.flashing:
                raise UARTError('Flash già in corso')
            self._close()
            self.flashing = True
        try:
            yield
        finally:
            with self.lock:
                self.flashing = False

    def _message(self):
        self.rx += self.serial.readline(16384)
        if len(self.rx) > 16384:
            self.rx = b''
            return None
        if not self.rx.endswith(b'\n'):
            return None
        line, self.rx = self.rx, b''
        try:
            message = json.loads(line.decode('utf-8'))
            return message if isinstance(message, dict) else None
        except (ValueError, UnicodeError):
            return None  # Ignore ROM boot messages and console text.

    def connect(self):
        with self.lock:
            if self.flashing:
                raise UARTError('Flash in corso: attendi il completamento prima di connetterti')
            self._close()
            try:
                self.serial = self.serial_factory(self.port, self.baud, timeout=.2, write_timeout=2)
                deadline, next_hello = time.monotonic() + 8, 0
                while time.monotonic() < deadline:
                    if time.monotonic() >= next_hello:
                        self.serial.write(b'HELLO\n')
                        next_hello = time.monotonic() + 1
                    message = self._message()
                    if message and message.get('type') == 'ready':
                        features = message.get('features')
                        if (not isinstance(message.get('job_id'), str) or not message['job_id'] or
                                type(features) is not int or not 1 <= features <= 4096):
                            raise UARTError('Identificazione firmware non valida')
                        self.board = message
                        return self.status()
                raise UARTError('La ESP32 non risponde a HELLO. Controlla porta, baud e firmware UART.', 504)
            except Exception as error:
                self._close()
                if isinstance(error, UARTError):
                    raise
                raise UARTError(f'Impossibile aprire la UART sul server: {error}', 503) from error

    def send(self, job_id, values):
        with self.lock:
            if self.flashing:
                raise UARTError('Flash in corso: riconnetti la ESP32 al termine')
            if self.serial is None or self.board is None:
                raise UARTError('Connetti la ESP32 dal pannello UART')
            if job_id != self.board['job_id']:
                raise UARTError('I campioni non appartengono al firmware caricato sulla ESP32')
            count = self.board['features']
            if (not isinstance(values, list) or len(values) != count or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                raise UARTError(f'Servono esattamente {count} valori grezzi finiti', 400)
            frame = (','.join(str(v) for v in values) + '\n').encode('ascii')
            if len(frame) >= count * 32 + 64:
                raise UARTError('Curva troppo lunga per il buffer del firmware', 400)
            try:
                self.serial.reset_input_buffer()
                self.rx = b''
                self.serial.write(frame)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    message = self._message()
                    if not message:
                        continue
                    if message.get('type') == 'ready':
                        self._close()
                        raise UARTError('La ESP32 si è riavviata: riconnettila')
                    if message.get('type') == 'error':
                        raise UARTError(message.get('message', 'Curva rifiutata dal firmware'), 400)
                    if message.get('type') == 'inference':
                        soh = message.get('soh')
                        if (message.get('job_id') != job_id or type(soh) not in (int, float) or
                                not math.isfinite(soh)):
                            self._close()
                            raise UARTError('Risposta firmware non valida: riconnetti la ESP32')
                        return message
                self._close()  # A late response must not be attributed to the next curve.
                raise UARTError('Timeout inferenza: riconnetti la ESP32', 504)
            except UARTError:
                raise
            except Exception as error:
                self._close()
                raise UARTError(f'Errore UART: {error}', 503) from error


def start_uart_server(bridge, port=8081):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._serve()

        def do_POST(self):
            self._serve()

        def _serve(self):
            try:
                payload = {}
                if self.command == 'POST':
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 <= length <= 131072:
                        raise UARTError('Richiesta troppo grande', 400)
                    payload = json.loads(self.rfile.read(length) or b'{}')
                    if not isinstance(payload, dict):
                        raise UARTError('Richiesta JSON non valida', 400)
                route = (self.command, self.path)
                if route == ('GET', '/status'):
                    result = bridge.status()
                elif route == ('POST', '/connect'):
                    result = bridge.connect()
                elif route == ('POST', '/disconnect'):
                    result = bridge.disconnect()
                elif route == ('POST', '/send'):
                    result = bridge.send(payload.get('job_id'), payload.get('values'))
                else:
                    raise UARTError('Endpoint UART non trovato', 404)
                self._reply(200, result)
            except UARTError as error:
                self._reply(error.status, {'detail': str(error)})
            except (ValueError, OverflowError):
                self._reply(400, {'detail': 'Richiesta JSON non valida'})

        def _reply(self, status, result):
            body = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name='uart-http')
    thread.start()
    return server
