import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Premium financial/AI palette.
        ink: {
          950: "#070a12",
          900: "#0b1020",
          800: "#111827",
          700: "#1c2333",
        },
        brand: {
          400: "#5eead4",
          500: "#2dd4bf",
          600: "#14b8a6",
        },
        accent: {
          400: "#818cf8",
          500: "#6366f1",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(45,212,191,0.15), 0 20px 60px -20px rgba(45,212,191,0.25)",
      },
      backgroundImage: {
        "radial-fade":
          "radial-gradient(60% 60% at 50% 0%, rgba(99,102,241,0.15) 0%, rgba(7,10,18,0) 70%)",
      },
    },
  },
  plugins: [],
};

export default config;
