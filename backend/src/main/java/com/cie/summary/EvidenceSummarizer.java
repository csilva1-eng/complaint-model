package com.cie.summary;

import com.cie.model.Models.*;

import java.util.List;

/**
 * Turns a cluster plus its representative complaints into a ticket-ready summary.
 *
 * Contract every implementation must honour:
 *   - observed facts (counts, dates, trend) come from the data, never from a model;
 *   - the hypothesis is labelled "Unverified" and is a lead for investigation, not a finding;
 *   - every cited complaint id belongs to the evidence that was supplied.
 */
public interface EvidenceSummarizer {
    Summary summarize(Cluster cluster, List<Complaint> evidence, List<TrendPoint> trend);
}
