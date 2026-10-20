// Client-side mirror of the Spring Boot API's JSON. Property names match Java's camelCase fields;
// System.Text.Json's web defaults handle the case difference.
namespace CIE.Web.Models;

public record NameShare(string Name, double Share);

public record Cluster(int ClusterId, int Size, string Label, List<NameShare> TopProducts,
                      List<NameShare> TopIssues, List<NameShare> TopSubIssues, List<NameShare> TopCompanies,
                      List<long> RepresentativeIds, string FirstDate, string LastDate);

public record TrendPoint(int ClusterId, string WeekStart, int Count, double BaselineMean,
                         double ZScore, bool Flagged);

public record Signal(int ClusterId, string WeekStart, int Count, double BaselineMean,
                     double ZScore, string Reason);

public record CompanySignal(string Company, string Tier, string WeekStart, int Count, double BaselineMean,
                             double ZScore, string Reason);

public record CompanyOption(string Name, long ComplaintCount);

public record CompanyCluster(int ClusterId, string Label, long ComplaintCount, int ClusterSize);

// The CFPB file has no free text, so a complaint is its structured fields.
// ProfileCount = how many complaints share this exact product, issue and company.
public record Complaint(long Id, string Date, string Product, string SubProduct, string Issue,
                        string SubIssue, string Company, string State, string SubmittedVia,
                        string CompanyResponse, string Timely, int ClusterId, int ProfileCount);

public record SimilarComplaint(Complaint Complaint, double Score);

public record ClusterDetail(Cluster Cluster, List<Complaint> Representatives, List<TrendPoint> Trend);

// Status is DRAFT, APPROVED, CREATED or REJECTED. Only a person moves a ticket between them.
public record Ticket(string Id, int ClusterId, string? Company, string Status, string Title, string ObservedPattern,
                     string Hypothesis, string NextSteps, List<long> EvidenceIds, string? JiraKey,
                     List<string> Audit);

public record TicketEdit(string Title, string ObservedPattern, string Hypothesis, string NextSteps);