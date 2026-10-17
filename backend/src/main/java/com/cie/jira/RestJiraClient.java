package com.cie.jira;

import com.cie.model.Models.Ticket;
import com.fasterxml.jackson.databind.JsonNode;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Map;

/**
 * Real Jira Cloud integration (cie.jira.mode=rest), using the v2 issue API with basic auth
 * (account email + API token).
 *
 * Production hardening to add: retry with backoff on 429/5xx, a rate limiter, an idempotency check
 * (search Jira for an existing issue carrying the cluster id before creating), and secrets from a
 * vault rather than config. The audit trail already lives on the Ticket.
 */
@Component
@ConditionalOnProperty(name = "cie.jira.mode", havingValue = "rest")
public class RestJiraClient implements JiraClient {
    private final RestClient http = RestClient.create();
    private final String baseUrl;
    private final String projectKey;
    private final String authHeader;

    public RestJiraClient(@Value("${cie.jira.base-url}") String baseUrl,
                          @Value("${cie.jira.email}") String email,
                          @Value("${cie.jira.api-token}") String apiToken,
                          @Value("${cie.jira.project-key}") String projectKey) {
        this.baseUrl = baseUrl.replaceAll("/+$", "");
        this.projectKey = projectKey;
        this.authHeader = "Basic " + Base64.getEncoder()
                .encodeToString((email + ":" + apiToken).getBytes(StandardCharsets.UTF_8));
    }

    @Override
    public String createIssue(Ticket t) {
        Map<String, Object> body = Map.of("fields", Map.of(
                "project", Map.of("key", projectKey),
                "summary", t.title(),
                "issuetype", Map.of("name", "Task"),
                "description", t.toPlainText()));

        JsonNode res = http.post().uri(baseUrl + "/rest/api/2/issue")
                .header("Authorization", authHeader)
                .contentType(MediaType.APPLICATION_JSON)
                .body(body)
                .retrieve().body(JsonNode.class);
        return res.path("key").asText();
    }
}
