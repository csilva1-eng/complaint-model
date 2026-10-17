package com.cie;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Complaint Intelligence Engine - backend.
 *
 * Role in the system: the Python pipeline discovers patterns offline and writes JSON artifacts;
 * this service serves them to the Blazor dashboard and owns the human-in-the-loop workflow:
 *
 *   artifacts -> ArtifactStore -> REST API -> evidence summary -> TicketService (review/approve) -> Jira
 *
 * Nothing here trains models. Nothing here creates a Jira ticket without an explicit approval call.
 */
@SpringBootApplication
public class CieApplication {
    public static void main(String[] args) {
        SpringApplication.run(CieApplication.class, args);
    }
}
