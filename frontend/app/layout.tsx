import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "DawaRx — AI Prescription Decipherer & Generic Finder",
  description:
    "Upload a photo of your handwritten Indian doctor's prescription. DawaRx uses AI to extract medicines and find affordable Jan Aushadhi generic alternatives, saving you money.",
  keywords: ["prescription", "generic medicine", "Jan Aushadhi", "PMBJP", "AI OCR", "India"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
