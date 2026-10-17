namespace CIE.Web.Services;

/// <summary>The backend answered with an error; Message is the reason it gave.</summary>
public class ApiException(string message) : Exception(message)
{
    /// <summary>Turns any failure into a sentence an analyst can act on.</summary>
    public static string Describe(Exception e) => e switch
    {
        ApiException => e.Message,
        HttpRequestException => "Can't reach the backend. Check that it is running on port 8080.",
        _ => "Something went wrong: " + e.Message,
    };
}
