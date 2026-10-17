package com.cie.jira;

import com.cie.model.Models.Ticket;

/**
 * The only door out of this service. TicketService calls it exclusively after a human approval.
 * Implementations return the created issue key (for example "CIE-101") or throw on failure.
 */
public interface JiraClient {
    String createIssue(Ticket ticket);
}
