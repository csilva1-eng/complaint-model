package com.cie.summary;

import com.cie.model.Models.*;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.Comparator;
import java.util.List;

/**
 * Default summarizer: deterministic, free, and offline. It also supplies the factual
 * "observed pattern" text that the LLM summarizer reuses, so numbers never come from a model.
 *
 * The CFPB file carries no complaint text, so everything here is about product, issue and company.
 * The hypothesis says so plainly and only distinguishes the two readings the data can support:
 * concentrated in one company (a company-specific cause is plausible) or spread across many
 * (an industry-wide or seasonal cause is plausible).
 */
@Component
@ConditionalOnProperty(name = "cie.summarizer", havingValue = "template", matchIfMissing = true)
public class TemplateEvidenceSummarizer implements EvidenceSummarizer {

    @Override
    public Summary summarize(Cluster c, List<Complaint> evidence, List<TrendPoint> trend) {
        NameShare issue = first(c.topIssues());
        NameShare subIssue = first(c.topSubIssues());
        NameShare company = first(c.topCompanies());

        StringBuilder pattern = new StringBuilder(
                "%d complaints between %s and %s fall into this theme. The most common issue is \"%s\" (%s of the theme)."
                        .formatted(c.size(), c.firstDate(), c.lastDate(), issue.name(), pct(issue.share())));
        if (!subIssue.name().equals("Unknown") && subIssue.share() > 0) {
            pattern.append(" The most common sub-issue is \"%s\" (%s).".formatted(subIssue.name(), pct(subIssue.share())));
        }
        pattern.append(" The company named most often is %s (%s).".formatted(company.name(), pct(company.share())));

        trend.stream().filter(TrendPoint::flagged)
                .max(Comparator.comparingDouble(TrendPoint::zScore))
                .ifPresentOrElse(
                        p -> pattern.append(" Strongest alert: week of %s had %d complaints against a trailing average of %.1f (z = %.1f)."
                                .formatted(p.weekStart(), p.count(), p.baselineMean(), p.zScore())),
                        () -> pattern.append(" No weekly volume alerts were raised for this theme."));

        String reading = company.share() >= 0.5
                ? "Complaints are concentrated in %s, which could point to a company-specific process or system change."
                        .formatted(company.name())
                : "Complaints are spread across many companies, which could point to an industry-wide, seasonal or regulatory cause.";
        String hypothesis = "Unverified: this theme is defined only by product, issue and company; no complaint text "
                + "was analysed, so the cause is unknown. " + reading
                + " Confirm with logs and reproduction before treating this as fact.";

        return new Summary(pattern.toString(), hypothesis, evidence.stream().map(Complaint::id).toList());
    }

    private static NameShare first(List<NameShare> shares) {
        return shares.isEmpty() ? new NameShare("Unknown", 0) : shares.get(0);
    }

    private static String pct(double share) {
        return Math.round(share * 100) + "%";
    }
}