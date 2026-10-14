#include <cstdio>
#include <cstring>
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_err.h"
#include "model_adapter.hpp"
#include "deployment.hpp"
#include "protocol.hpp"
#include "led.hpp"

static constexpr uart_port_t PORT = static_cast<uart_port_t>(UART_PORT);
static char line[FEATURE_COUNT * 32 + 64];
static double values[FEATURE_COUNT];
static int8_t input[FEATURE_COUNT];

static void reply(const char* message) {
    uart_write_bytes(PORT, message, std::strlen(message));
}
static void process_line() {
    if (std::strcmp(line, "HELLO") == 0) {
        char result[160];
        std::snprintf(result, sizeof(result),
            "{\"type\":\"ready\",\"job_id\":\"%s\",\"features\":%u,\"threshold\":80}\n",
            JOB_ID, static_cast<unsigned>(FEATURE_COUNT));
        reply(result);
        return;
    }
    if (!parse_features(line, values, FEATURE_COUNT)) {
        clear_led();
        reply("{\"type\":\"error\",\"message\":\"Expected a complete CSV curve of finite raw features\"}\n");
        return;
    }
    unsigned clipped = 0;
    for (size_t i = 0; i < FEATURE_COUNT; ++i) {
        bool saturated = false;
        if (!quantize_feature(values[i], SCALE[i], OFFSET[i], INPUT_MULTIPLIER, input[i], saturated)) {
            clear_led();
            reply("{\"type\":\"error\",\"message\":\"Preprocessing overflow\"}\n");
            return;
        }
        clipped += saturated;
    }
    const float soh = model_soh(input);
    if (!std::isfinite(soh)) {
        clear_led();
        reply("{\"type\":\"error\",\"message\":\"Non-finite model output\"}\n");
        return;
    }
    const bool degraded = is_degraded(soh);
    set_led(degraded ? LED_BRIGHTNESS : 0, degraded ? 0 : LED_BRIGHTNESS, 0);
    char result[200];
    std::snprintf(result, sizeof(result),
        "{\"type\":\"inference\",\"job_id\":\"%s\",\"soh\":%.6f,\"color\":\"%s\",\"clipped_features\":%u}\n",
        JOB_ID, static_cast<double>(soh), degraded ? "red" : "green", clipped);
    reply(result);
}
extern "C" void app_main(void) {
    configure_led();
    uart_config_t config = {};
    config.baud_rate = UART_BAUD;
    config.data_bits = UART_DATA_8_BITS;
    config.parity = UART_PARITY_DISABLE;
    config.stop_bits = UART_STOP_BITS_1;
    config.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
    config.source_clk = UART_SCLK_DEFAULT;
    ESP_ERROR_CHECK(uart_param_config(PORT, &config));
    ESP_ERROR_CHECK(uart_set_pin(PORT, UART_TX_GPIO, UART_RX_GPIO, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(PORT, sizeof(line) * 2, 0, 0, nullptr, 0));
    std::strcpy(line, "HELLO");
    process_line();
    size_t used = 0;
    bool dropping = false;
    uint8_t chunk[256];
    while (true) {
        int received = uart_read_bytes(PORT, chunk, sizeof(chunk), pdMS_TO_TICKS(100));
        for (int i = 0; i < received; ++i) {
            const uint8_t ch = chunk[i];
            if (ch == '\n') {
                if (dropping) {
                    clear_led();
                    reply("{\"type\":\"error\",\"message\":\"Invalid or oversized frame\"}\n");
                } else if (used) {
                    line[used] = '\0';
                    process_line();
                }
                used = 0;
                dropping = false;
            } else if (!dropping) {
                if (ch == 0 || used >= sizeof(line) - 1) dropping = true;
                else line[used++] = static_cast<char>(ch);
            }
        }
        // Allow the idle task/watchdog to run even under sustained traffic.
        vTaskDelay(1);
    }
}
