#pragma once
#include <cstdint>
#include "driver/rmt_tx.h"
#include "driver/rmt_encoder.h"
#include "esp_err.h"
#include "deployment.hpp"

// One onboard WS2812 on the FireBeetle ESP32-E: GRB, MSB first.
// A full pixel plus the reset pulse fits in a single 64-symbol RMT block.
static rmt_channel_handle_t led_channel = nullptr;
static rmt_encoder_handle_t led_encoder = nullptr;

static void set_led(uint8_t red, uint8_t green, uint8_t blue) {
    const uint8_t grb[] = {green, red, blue};
    rmt_symbol_word_t frame[25] = {};
    for (unsigned bit = 0; bit < 24; ++bit) {
        const bool one = grb[bit / 8] & (0x80 >> (bit % 8));
        frame[bit].level0 = 1;
        frame[bit].duration0 = one ? 8 : 4; // 0.8/0.4 us at 10 MHz
        frame[bit].level1 = 0;
        frame[bit].duration1 = one ? 4 : 8;
    }
    // Hold low for 300 us, including WS2812 revisions with longer reset times.
    frame[24].duration0 = 1500;
    frame[24].duration1 = 1500;
    rmt_transmit_config_t transmit = {};
    ESP_ERROR_CHECK(rmt_transmit(led_channel, led_encoder, frame, sizeof(frame), &transmit));
    // Keep the stack-backed frame alive until hardware has sent it.
    ESP_ERROR_CHECK(rmt_tx_wait_all_done(led_channel, 1000));
}

static void clear_led() { set_led(0, 0, 0); }

static void configure_led() {
    rmt_tx_channel_config_t channel = {};
    channel.gpio_num = static_cast<gpio_num_t>(LED_RGB_GPIO);
    channel.clk_src = RMT_CLK_SRC_DEFAULT;
    channel.resolution_hz = 10000000;
    channel.mem_block_symbols = 64;
    channel.trans_queue_depth = 1;
    ESP_ERROR_CHECK(rmt_new_tx_channel(&channel, &led_channel));
    rmt_copy_encoder_config_t encoder = {};
    ESP_ERROR_CHECK(rmt_new_copy_encoder(&encoder, &led_encoder));
    ESP_ERROR_CHECK(rmt_enable(led_channel));
    clear_led();
}
