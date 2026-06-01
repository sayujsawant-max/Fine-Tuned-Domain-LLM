import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FinSage-7B — Fine-tuned Financial Filing LLM",
  description:
    "Fine-tuned Mistral-7B for financial filing analysis. Paste a filing excerpt, ask a question, and get a grounded answer — a recruiter-friendly demo.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen font-sans">{children}</body>
    </html>
  );
}
