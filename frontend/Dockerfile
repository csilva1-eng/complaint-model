FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY CIE.Web/*.csproj CIE.Web/
RUN dotnet restore CIE.Web
COPY . .
RUN dotnet publish CIE.Web/*.csproj -c Release -o /app --no-restore

FROM nginx:alpine
COPY --from=build /app/wwwroot /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80