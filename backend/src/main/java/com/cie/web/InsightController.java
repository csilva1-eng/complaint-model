package com.cie.web;

import com.cie.model.Models.*;
import com.cie.store.ArtifactStore;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.Locale;
import java.util.Optional;

/**
 * Read-only endpoints behind the dashboard: what is rising, what the themes are, and which
 * complaints read alike.
 *
 *   GET /api/signals?limit=        flagged cluster-weeks, strongest first
 *   GET /api/clusters              all themes
 *   GET /api/clusters/{id}         one theme: representative complaints + weekly trend
 *   GET /api/complaints/search?q=  text match on company, product, issue and sub-issue
 *   GET /api/complaints/{id}/similar?k=   complaints with the most similar product/issue/company
 *                                          profile (cosine similarity, precomputed with FAISS)
 */
@RestController
@RequestMapping("/api")
public class InsightController {
    private final ArtifactStore store;

    public InsightController(ArtifactStore store) {
        this.store = store;
    }

    @GetMapping("/signals")
    public List<Signal> signals(@RequestParam(defaultValue = "20") int limit) {
        return store.signals().stream().limit(Math.max(1, Math.min(limit, 200))).toList();
    }

    @GetMapping("/clusters")
    public List<Cluster> clusters() {
        return store.clusters();
    }

    @GetMapping("/clusters/{id}")
    public ClusterDetail cluster(@PathVariable int id) {
        Cluster c = store.cluster(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "No cluster " + id));
        List<Complaint> reps = c.representativeIds().stream()
                .map(store::complaint).flatMap(Optional::stream).toList();
        return new ClusterDetail(c, reps, store.trend(id));
    }

    @GetMapping("/complaints/search")
    public List<Complaint> search(@RequestParam String q, @RequestParam(defaultValue = "25") int limit) {
        String needle = q.toLowerCase(Locale.ROOT).trim();
        if (needle.isEmpty()) return List.of();
        return store.allComplaints().stream()
                .filter(c -> searchable(c).contains(needle))
                .limit(Math.max(1, Math.min(limit, 100)))
                .toList();
    }

    private static String searchable(Complaint c) {
        return String.join(" ", c.company(), c.product(), c.subProduct(), c.issue(), c.subIssue())
                .toLowerCase(Locale.ROOT);
    }

    @GetMapping("/complaints/{id}/similar")
    public List<SimilarComplaint> similar(@PathVariable long id, @RequestParam(defaultValue = "10") int k) {
        if (store.complaint(id).isEmpty()) {
            throw new ResponseStatusException(HttpStatus.NOT_FOUND, "No complaint " + id);
        }
        return store.neighbors(id).stream().limit(Math.max(1, Math.min(k, 50)))
                .map(n -> store.complaint(n.id()).map(c -> new SimilarComplaint(c, n.score())))
                .flatMap(Optional::stream)
                .toList();
    }
}