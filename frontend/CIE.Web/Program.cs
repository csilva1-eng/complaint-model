// Blazor WebAssembly entry point.
//
// The dashboard is a static client app: all data comes from the Spring Boot API, whose base URL
// is read from wwwroot/appsettings.json. Run it with:  dotnet run --urls http://localhost:5173
using CIE.Web;
using CIE.Web.Services;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

var apiBase = builder.Configuration["ApiBaseUrl"] ?? "http://localhost:8081/";
builder.Services.AddScoped(_ => new HttpClient { BaseAddress = new Uri(apiBase) });
builder.Services.AddScoped<CieApi>();

await builder.Build().RunAsync();
