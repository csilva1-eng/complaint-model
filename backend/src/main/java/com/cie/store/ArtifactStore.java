package com.cie.store;

import com.cie.model.Models.*;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Read-only, in-memory view of the pipeline's JSON artifacts, loaded once at startup.
 *
 * Why in memory: the prototype corpus is tens of thousands of rows, so a lookup map beats
 * running a database. If the corpus outgrows RAM, this class is the seam to replace with
 * PostgreSQL (+ pgvector) without touching controllers or services.
 *
 * Re-run the Python pipeline, then restart the backend to pick up new artifacts.
 */
@Component
public class ArtifactStore {
    private final Path dir;
    private final ObjectMapper mapper;

    private Map<Long, Complaint> complaints;
    private List<Cluster> clusters;
    private Map<Integer, Cluster> clusterById;
    private Map<Integer, List<TrendPoint>> trendByCluster;
    private List<Signal> signals;
    private Map<Long, List<Neighbor>> neighbors;

    public ArtifactStore(@Value("${cie.artifacts-dir}") String dir, ObjectMapper mapper) {
        this.dir = Path.of(dir);
        this.mapper = mapper;
    }

    @PostConstruct
    void load() throws IOException {
        complaints = read("complaints.json", new TypeReference<List<Complaint>>() {}).stream()
                .collect(Collectors.toMap(Complaint::id, c -> c, (a, b) -> a, LinkedHashMap::new));

        clusters = read("clusters.json", new TypeReference<List<Cluster>>() {});
        clusterById = clusters.stream().collect(Collectors.toMap(Cluster::clusterId, c -> c));

        trendByCluster = read("trends.json", new TypeReference<List<TrendPoint>>() {}).stream()
                .collect(Collectors.groupingBy(TrendPoint::clusterId, Collectors.toCollection(ArrayList::new)));
        trendByCluster.values().forEach(list -> list.sort(Comparator.comparing(TrendPoint::weekStart)));

        signals = new ArrayList<>(read("signals.json", new TypeReference<List<Signal>>() {}));
        signals.sort(Comparator.comparingDouble(Signal::zScore).reversed());

        neighbors = read("neighbors.json", new TypeReference<Map<Long, List<Neighbor>>>() {});
    }

    private <T> T read(String file, TypeReference<T> type) throws IOException {
        Path p = dir.resolve(file);
        if (!Files.exists(p)) {
            throw new IllegalStateException("Missing " + p.toAbsolutePath()
                    + ". Run the ML pipeline first: cd ml && python run_pipeline.py");
        }
        return mapper.readValue(p.toFile(), type);
    }

    public Optional<Complaint> complaint(long id) { return Optional.ofNullable(complaints.get(id)); }
    public Collection<Complaint> allComplaints() { return complaints.values(); }
    public List<Cluster> clusters() { return clusters; }
    public Optional<Cluster> cluster(int id) { return Optional.ofNullable(clusterById.get(id)); }
    public List<TrendPoint> trend(int clusterId) { return trendByCluster.getOrDefault(clusterId, List.of()); }
    public List<Signal> signals() { return signals; }
    public List<Neighbor> neighbors(long id) { return neighbors.getOrDefault(id, List.of()); }
}
