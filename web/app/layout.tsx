import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Vessel Vision Inference Hub",
  description:
    "Run and compare 11 vessel-detection models side-by-side on a single image.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
