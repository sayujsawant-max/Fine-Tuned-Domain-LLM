/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Only safe, public values are exposed to the browser. The backend API
  // secret is read server-side in app/api/* routes and is never bundled here.
  env: {
    NEXT_PUBLIC_API_BASE_URL:
      process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8080/v1",
    NEXT_PUBLIC_DEMO_MODE: process.env.NEXT_PUBLIC_DEMO_MODE ?? "true",
    NEXT_PUBLIC_APP_NAME: process.env.NEXT_PUBLIC_APP_NAME ?? "FinSage-7B",
  },
};

export default nextConfig;
