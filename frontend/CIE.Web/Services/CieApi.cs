using System.Net.Http.Json;
using CIE.Web.Models;

namespace CIE.Web.Services;

/// <summary>
/// The dashboard's only gateway to the backend: one method per REST endpoint, no UI logic.
/// Reads are plain GETs; ticket changes go through <see cref="Send{T}"/> so error reasons
/// from the server (for example "Only drafts can be edited") reach the screen.
/// </summary>
public class CieApi(HttpClient http)
{
    // ---- insight (read-only) -----------------------------------------------------------------
    public async Task<List<Signal>> GetSignals(int limit = 25) =>
        await http.GetFromJsonAsync<List<Signal>>($"api/signals?limit={limit}") ?? [];

    public async Task<List<CompanyOption>> GetCompanies() =>
        await http.GetFromJsonAsync<List<CompanyOption>>("api/companies") ?? [];

    public async Task<List<CompanySignal>> GetCompanySignals(string company) =>
        await http.GetFromJsonAsync<List<CompanySignal>>($"api/companies/signals?company={Uri.EscapeDataString(company)}") ?? [];

    public async Task<List<CompanyCluster>> GetCompanyClusters(string company) =>
        await http.GetFromJsonAsync<List<CompanyCluster>>($"api/companies/clusters?company={Uri.EscapeDataString(company)}") ?? [];

    public async Task<List<Cluster>> GetClusters() =>
        await http.GetFromJsonAsync<List<Cluster>>("api/clusters") ?? [];

    public async Task<ClusterDetail> GetCluster(int id) =>
        (await http.GetFromJsonAsync<ClusterDetail>($"api/clusters/{id}"))!;

    public async Task<List<Complaint>> SearchComplaints(string query, string company) =>
        await http.GetFromJsonAsync<List<Complaint>>(
            $"api/complaints/search?q={Uri.EscapeDataString(query)}&company={Uri.EscapeDataString(company)}") ?? [];

    public async Task<List<SimilarComplaint>> GetSimilar(long id) =>
        await http.GetFromJsonAsync<List<SimilarComplaint>>($"api/complaints/{id}/similar") ?? [];

    // ---- ticket workflow ---------------------------------------------------------------------
    public Task<Ticket> CreateDraft(int clusterId) => Send<Ticket>(HttpMethod.Post, $"api/tickets?clusterId={clusterId}");
    public async Task<List<Ticket>> GetTickets(string? company = null) =>
        await http.GetFromJsonAsync<List<Ticket>>(company is null
            ? "api/tickets"
            : $"api/tickets?company={Uri.EscapeDataString(company)}") ?? [];
    public Task<Ticket> CreateCompanyDraft(string company, string weekStart) =>
        Send<Ticket>(HttpMethod.Post,
            $"api/tickets/company?company={Uri.EscapeDataString(company)}&weekStart={Uri.EscapeDataString(weekStart)}");
    public Task<Ticket> GetTicket(string id) => Send<Ticket>(HttpMethod.Get, $"api/tickets/{id}");
    public Task<Ticket> UpdateTicket(string id, TicketEdit edit) => Send<Ticket>(HttpMethod.Put, $"api/tickets/{id}", edit);
    public Task<Ticket> Approve(string id) => Send<Ticket>(HttpMethod.Post, $"api/tickets/{id}/approve");
    public Task<Ticket> Reject(string id) => Send<Ticket>(HttpMethod.Post, $"api/tickets/{id}/reject");

    private async Task<T> Send<T>(HttpMethod method, string url, object? body = null)
    {
        using var request = new HttpRequestMessage(method, url);
        if (body is not null) request.Content = JsonContent.Create(body);

        using var response = await http.SendAsync(request);
        if (!response.IsSuccessStatusCode)
        {
            ErrorBody? problem = null;
            try { problem = await response.Content.ReadFromJsonAsync<ErrorBody>(); } catch { /* not JSON */ }
            throw new ApiException(string.IsNullOrWhiteSpace(problem?.Message)
                ? $"The backend returned an error ({(int)response.StatusCode})."
                : problem!.Message!);
        }
        return (await response.Content.ReadFromJsonAsync<T>())!;
    }

    private record ErrorBody(string? Message);
}
