package com.cie.model;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Every data shape in one place.
 *
 * Section 1 mirrors the JSON written by the Python pipeline (field names must match, camelCase).
 * Section 2 is what the REST API returns to the dashboard.
 * Section 3 is the ticket workflow, which lives only in this service.
 */
public final class Models {
    private Models() {}

    // ---- 1. Pipeline artifacts ---------------------------------------------------------------
    /**
     * One complaint's structured fields. The CFPB file has no free text, so a complaint is described
     * by what it is about. profileCount = how many complaints share this exact product, issue and company.
     */
    public record Complaint(long id, String date, String product, String subProduct, String issue,
                            String subIssue, String company, String state, String submittedVia,
                            String companyResponse, String timely, int clusterId, int profileCount) {}

    public record NameShare(String name, double share) {}

    /** A theme: a recurring mix of product, issue and company, described by its biggest members. */
    public record Cluster(int clusterId, int size, String label, List<NameShare> topProducts,
                          List<NameShare> topIssues, List<NameShare> topSubIssues,
                          List<NameShare> topCompanies, List<Long> representativeIds,
                          String firstDate, String lastDate) {}

    /** One cluster-week: observed count versus its trailing baseline. */
    public record TrendPoint(int clusterId, String weekStart, int count, double baselineMean,
                             double zScore, boolean flagged) {}

    /** A flagged cluster-week, with a plain-language explanation of why it was flagged. */
    public record Signal(int clusterId, String weekStart, int count, double baselineMean,
                         double zScore, String reason) {}

    public record CompanySignal(String company, String tier, String weekStart, int count, double baselineMean,
                                double zScore, String reason) {}

    public record Neighbor(long id, double score) {}

    // ---- 2. API responses --------------------------------------------------------------------
    public record ClusterDetail(Cluster cluster, List<Complaint> representatives, List<TrendPoint> trend) {}

    public record SimilarComplaint(Complaint complaint, double score) {}

    // ---- 3. Ticket workflow ------------------------------------------------------------------
    /** DRAFT -> APPROVED -> CREATED, or REJECTED. A person triggers every transition. */
    public enum TicketStatus { DRAFT, APPROVED, CREATED, REJECTED }

    /** Output of a summarizer: facts from the data, a labelled hypothesis, and the evidence it cites. */
    public record Summary(String observedPattern, String hypothesis, List<Long> citedComplaintIds) {}

    /** Fields an analyst may change while the ticket is still a draft. */
    public record TicketEdit(String title, String observedPattern, String hypothesis, String nextSteps) {}

    /** Immutable; each change produces a new copy with an audit line appended. */
    public record Ticket(String id, int clusterId, String company, TicketStatus status, String title,
                         String observedPattern, String hypothesis, String nextSteps,
                         List<Long> evidenceIds, String jiraKey, List<String> audit) {

        public static String stamp(String note) {
            return Instant.now() + "  " + note;
        }

        public Ticket edited(TicketEdit e) {
                return new Ticket(id, clusterId, company, status, e.title(), e.observedPattern(), e.hypothesis(),
                    e.nextSteps(), evidenceIds, jiraKey, appended("Edited by analyst"));
        }

        public Ticket transition(TicketStatus next, String key, String note) {
            return new Ticket(id, clusterId, company, next, title, observedPattern, hypothesis, nextSteps,
                    evidenceIds, key != null ? key : jiraKey, appended(note));
        }

        /** Jira description. Evidence is IDs only, so no complaint text leaves this service. */
        public String toPlainText() {
            return String.join("\n\n",
                    observedPattern,
                    "Evidence (CFPB complaint IDs): " + evidenceIds,
                    "Hypothesis: " + hypothesis,
                    "Suggested next steps: " + nextSteps,
                    "Drafted by Complaint Intelligence Engine and approved by a human reviewer.");
        }

        private List<String> appended(String note) {
            List<String> out = new ArrayList<>(audit);
            out.add(stamp(note));
            return List.copyOf(out);
        }
    }
}