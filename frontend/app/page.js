/**
 * Main Page Component for VectorDB UI
 * Features: Add text, Semantic Search, Live Stats, and Result Cards.
 */

"use client";

import { useState, useEffect } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export default function Home() {
  // ---------- State Variables ----------
  const [addText, setAddText] = useState("");
  const [addMetadata, setAddMetadata] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [totalVectors, setTotalVectors] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [notification, setNotification] = useState({ message: "", type: "" });

  // ---------- Fetch Stats on Load ----------
  useEffect(() => {
    fetchStats();
  }, []);

  // Fetch total number of vectors from the backend
  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      const data = await response.json();
      setTotalVectors(data.total_vectors);
    } catch (error) {
      console.error("Failed to fetch stats:", error);
      setTotalVectors(0);
    }
  };

  // ---------- Helper: Show Notification ----------
  const showNotification = (message, type = "success") => {
    setNotification({ message, type });
    setTimeout(() => setNotification({ message: "", type: "" }), 4000);
  };

  // ---------- Handle: Add Text ----------
  const handleAddText = async (e) => {
    e.preventDefault();
    if (!addText.trim()) {
      showNotification("Please enter some text to add.", "error");
      return;
    }

    setIsLoading(true);
    try {
      const payload = {
        text: addText,
        metadata: addMetadata || "User Input",
      };

      const response = await fetch(`${API_BASE_URL}/add_text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.ok) {
        showNotification(`✅ Text added successfully! ID: ${data.id || "N/A"}`, "success");
        setAddText("");
        setAddMetadata("");
        fetchStats(); // Update the vector count
      } else {
        showNotification(`❌ Error: ${data.detail || "Something went wrong"}`, "error");
      }
    } catch (error) {
      showNotification("❌ Failed to connect to the backend.", "error");
    } finally {
      setIsLoading(false);
    }
  };

  // ---------- Handle: Search Text ----------
  const handleSearch = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      showNotification("Please enter a search query.", "error");
      return;
    }

    setIsSearching(true);
    setSearchResults([]);
    try {
      const payload = {
        text: searchQuery,
        top_k: 5,
      };

      const response = await fetch(`${API_BASE_URL}/search_text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.ok) {
        setSearchResults(data.results || []);
        if (data.results.length === 0) {
          showNotification("No similar vectors found.", "info");
        } else {
          showNotification(`Found ${data.results.length} results!`, "success");
        }
      } else {
        showNotification(`❌ Error: ${data.detail || "Search failed"}`, "error");
      }
    } catch (error) {
      showNotification("❌ Failed to connect to the backend.", "error");
    } finally {
      setIsSearching(false);
    }
  };

  // ---------- Render UI ----------
  return (
    <main className="min-h-screen p-4 md:p-8 flex flex-col items-center justify-start">
      {/* Notification Toast */}
      {notification.message && (
        <div
          className={`fixed top-6 left-1/2 -translate-x-1/2 z-50 px-6 py-3 rounded-xl shadow-2xl transition-all duration-500 ${
            notification.type === "error"
              ? "bg-red-500/90 text-white"
              : notification.type === "info"
              ? "bg-blue-500/90 text-white"
              : "bg-green-500/90 text-white"
          } backdrop-blur-md border border-white/20`}
        >
          {notification.message}
        </div>
      )}

      {/* Main Container */}
      <div className="w-full max-w-4xl">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-2 tracking-tight">
            🔍 Vector<span className="text-gradient">DB</span>
          </h1>
          <p className="text-white/60 text-lg font-light">
            Semantic Search Engine • AI-powered • Meaning, not just keywords
          </p>
          <div className="mt-4 inline-flex items-center gap-3 px-4 py-2 glass-card rounded-full text-white/80 text-sm">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
            Database Status:{" "}
            <span className="font-semibold text-white">{totalVectors} vectors stored</span>
          </div>
        </div>

        {/* Two Column Layout: Add + Search */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* ---------- Left Column: Add Text ---------- */}
          <div className="glass-card rounded-2xl p-6 text-white">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="text-2xl">📝</span> Add Text
            </h2>
            <form onSubmit={handleAddText} className="space-y-4">
              <div>
                <label className="block text-white/70 text-sm mb-1">Text Content</label>
                <textarea
                  value={addText}
                  onChange={(e) => setAddText(e.target.value)}
                  placeholder="e.g., Biryani is a spicy rice dish..."
                  className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-purple-400 transition-all"
                  rows="3"
                />
              </div>
              <div>
                <label className="block text-white/70 text-sm mb-1">Metadata (Optional)</label>
                <input
                  value={addMetadata}
                  onChange={(e) => setAddMetadata(e.target.value)}
                  placeholder="e.g., Pakistani Food"
                  className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-purple-400 transition-all"
                />
              </div>
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-xl transition-all shadow-lg hover:shadow-purple-500/25 flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <span className="spinner w-5 h-5 border-2 border-white border-t-transparent rounded-full"></span>
                    Adding...
                  </>
                ) : (
                  "🚀 Add to Database"
                )}
              </button>
            </form>
          </div>

          {/* ---------- Right Column: Search ---------- */}
          <div className="glass-card rounded-2xl p-6 text-white">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="text-2xl">🔎</span> Search
            </h2>
            <form onSubmit={handleSearch} className="space-y-4">
              <div>
                <label className="block text-white/70 text-sm mb-1">Your Query</label>
                <input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="e.g., I want something sweet..."
                  className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-purple-400 transition-all"
                />
              </div>
              <button
                type="submit"
                disabled={isSearching}
                className="w-full bg-white/20 hover:bg-white/30 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 rounded-xl transition-all border border-white/20 backdrop-blur-sm flex items-center justify-center gap-2"
              >
                {isSearching ? (
                  <>
                    <span className="spinner w-5 h-5 border-2 border-white border-t-transparent rounded-full"></span>
                    Searching...
                  </>
                ) : (
                  "✨ Search Semantics"
                )}
              </button>
            </form>
          </div>
        </div>

        {/* ---------- Results Section ---------- */}
        {searchResults.length > 0 && (
          <div className="mt-8 glass-card rounded-2xl p-6 text-white">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              📊 Results ({searchResults.length})
            </h3>
            <div className="space-y-3">
              {searchResults.map((item, index) => (
                <div
                  key={index}
                  className="bg-white/5 hover:bg-white/10 rounded-xl p-4 border border-white/10 transition-all"
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <p className="text-white/90 font-medium">
                        #{index + 1} · {item.id || "N/A"}
                      </p>
                      <p className="text-white/60 text-sm mt-1">
                        {item.metadata || "No metadata"}
                      </p>
                    </div>
                    <div className="ml-4 flex flex-col items-end">
                      <span className="text-sm font-bold text-green-300 bg-green-500/20 px-3 py-1 rounded-full">
                        {(item.similarity * 100).toFixed(1)}% Match
                      </span>
                    </div>
                  </div>
                  {/* Progress bar for similarity */}
                  <div className="mt-2 w-full h-1 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-green-400 to-purple-500 rounded-full transition-all duration-500"
                      style={{ width: `${item.similarity * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Empty state */}
        {searchResults.length === 0 && !isSearching && (
          <div className="mt-8 text-center text-white/40 text-sm">
            <p>Search results will appear here.</p>
            <p className="text-white/20">Try searching for "spicy", "sweet", or "dessert"!</p>
          </div>
        )}
      </div>
    </main>
  );
}