/**
 * Root Layout for the VectorDB UI
 * Sets up the global HTML structure, fonts, and metadata.
 */

import "./globals.css";

export const metadata = {
  title: "VectorDB - Semantic Search Engine",
  description:
    "A high-performance vector database with AI text embeddings. Search by meaning, not just keywords!",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-800 to-pink-700">
        {children}
      </body>
    </html>
  );
}