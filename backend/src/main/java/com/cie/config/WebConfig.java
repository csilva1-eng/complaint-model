package com.cie.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/** Lets the Blazor WebAssembly app (a different origin in development) call the API. */
@Configuration
public class WebConfig implements WebMvcConfigurer {
    private final String[] origins;

    public WebConfig(@Value("${cie.allowed-origins}") String[] origins) {
        this.origins = origins;
    }

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**").allowedOrigins(origins).allowedMethods("GET", "POST", "PUT");
    }
}
