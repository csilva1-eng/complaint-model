package com.cie.jira;

import com.cie.model.Models.Ticket;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.concurrent.atomic.AtomicInteger;

/** Demo mode (default): runs the full approval flow and logs the payload instead of calling Jira. */
@Component
@ConditionalOnProperty(name = "cie.jira.mode", havingValue = "mock", matchIfMissing = true)
public class MockJiraClient implements JiraClient {
    private static final Logger log = LoggerFactory.getLogger(MockJiraClient.class);
    private final AtomicInteger next = new AtomicInteger(100);

    @Override
    public String createIssue(Ticket t) {
        String key = "CIE-" + next.incrementAndGet();
        log.info("[mock Jira] would create {}: {}\n{}", key, t.title(), t.toPlainText());
        return key;
    }
}
