import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { ArrowLeft, Download } from "lucide-react";
import PageLayout from "@/components/layout/PageLayout";

interface BackendResponse {
  msme_category: string;
  eligible_schemes: {
    scheme: string;
    reason: string;
  }[];
}

const MASTER_SCHEMES = [
  "PM Mudra Yojana",
  "Stand-Up India",
  "CGTMSE",
  "Startup India Seed Fund Scheme",
  "Credit Linked Capital Subsidy Scheme",
  "Export Promotion Capital Goods Scheme",
  "North East Industrial Development Scheme",
  "Aspirational District Programme",
  "Mahila Udyam Nidhi Scheme",
  "PMEGP",
  "MSME Sustainable Certification",
  "Technology Upgradation Fund"
];

const fadeUp = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 }
};

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const [result, setResult] = useState<BackendResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [animatedPercent, setAnimatedPercent] = useState(0);

  // ---------------- SAFE FORM DATA ----------------
  const formData = useMemo(() => {
    if (location.state) return location.state;
    try {
      const stored = localStorage.getItem("predictionData");
      if (stored) return JSON.parse(stored);
    } catch {
      return null;
    }
    return null;
  }, [location.state]);

  // ---------------- FETCH BACKEND ----------------
  useEffect(() => {
    if (!formData) {
      navigate("/eligibility", { replace: true });
      return;
    }

    const fetchPrediction = async () => {
      try {
        const res = await fetch("https://msme-g3ht.onrender.com/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        });

        if (!res.ok) throw new Error("Backend error");

        const data = await res.json();
        setResult(data);
      } catch (error) {
        console.error("Prediction error:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchPrediction();
  }, [formData, navigate]);

  // ---------------- CALCULATIONS (SAFE) ----------------
  const totalSchemes = MASTER_SCHEMES.length;
  const eligibleCount = result?.eligible_schemes?.length || 0;
  const coveragePercentage =
    totalSchemes > 0
      ? Math.round((eligibleCount / totalSchemes) * 100)
      : 0;

  const strengthIndex = result
    ? Math.min(
        100,
        Math.round(
          coveragePercentage * 0.7 +
            (result.msme_category === "Micro"
              ? 20
              : result.msme_category === "Small"
              ? 15
              : 10)
        )
      )
    : 0;

  // ---------------- ANIMATION (HOOK SAFE) ----------------
  useEffect(() => {
    if (!result) return;

    let start = 0;
    const end = coveragePercentage;
    const duration = 800;
    const increment = end / (duration / 16);

    const counter = setInterval(() => {
      start += increment;
      if (start >= end) {
        start = end;
        clearInterval(counter);
      }
      setAnimatedPercent(Math.floor(start));
    }, 16);

    return () => clearInterval(counter);
  }, [coveragePercentage, result]);

  // ---------------- SAFE LOADING ----------------
  if (loading) {
    return (
      <PageLayout>
        <div className="flex justify-center items-center h-64">
          <p className="text-muted-foreground">Analyzing eligibility...</p>
        </div>
      </PageLayout>
    );
  }

  if (!result) {
    return (
      <PageLayout>
        <div className="flex justify-center items-center h-64">
          <p className="text-destructive">
            Unable to fetch eligibility result.
          </p>
        </div>
      </PageLayout>
    );
  }

  // ---------------- DOWNLOAD REPORT ----------------
  const downloadReport = () => {
    const content = `
CREDITINTEL MSME ELIGIBILITY REPORT

MSME Category: ${result.msme_category}
Coverage Score: ${coveragePercentage}%
Strength Index: ${strengthIndex}/100

Submitted Details:
${Object.entries(formData || {})
  .map(([k, v]) => `${k}: ${v}`)
  .join("\n")}

Eligible Schemes:
${result.eligible_schemes
  .map((s, i) => `${i + 1}. ${s.scheme} - ${s.reason}`)
  .join("\n")}
`;

    const blob = new Blob([content], {
      type: "text/plain;charset=utf-8;",
    });

    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "CreditIntel_Report.txt";
    link.click();
  };

  return (
    <PageLayout>
      <section className="py-16 lg:py-20">
        <div className="container mx-auto px-4 lg:px-8 max-w-5xl space-y-10">

          {/* MSME CATEGORY */}
          <motion.div {...fadeUp} className="bg-card rounded-3xl p-8 text-center">
            <h2 className="text-xl font-semibold mb-4">
              MSME Classification
            </h2>
            <p className="text-3xl font-bold text-primary">
              {result.msme_category}
            </p>
          </motion.div>

          {/* COVERAGE SCORE */}
          <motion.div {...fadeUp} className="bg-card rounded-3xl p-8 text-center">
            <h2 className="text-xl font-semibold mb-6">
              Scheme Coverage Score
            </h2>

            <div className="relative w-48 h-48 mx-auto flex items-center justify-center">
              <svg width="180" height="180">
                <circle
                  cx="90"
                  cy="90"
                  r="70"
                  stroke="hsl(var(--muted))"
                  strokeWidth="14"
                  fill="none"
                />
                <circle
                  cx="90"
                  cy="90"
                  r="70"
                  stroke="hsl(var(--primary))"
                  strokeWidth="14"
                  fill="none"
                  strokeDasharray={440}
                  strokeDashoffset={440 - (440 * animatedPercent) / 100}
                  strokeLinecap="round"
                  style={{
                    transition: "stroke-dashoffset 0.6s ease",
                  }}
                />
              </svg>

              <div className="absolute text-center">
                <p className="text-3xl font-bold">
                  {animatedPercent}%
                </p>
                <p className="text-xs text-muted-foreground">
                  Coverage
                </p>
              </div>
            </div>

            <div className="mt-6">
              <p className="text-sm text-muted-foreground">
                MSME Strength Index
              </p>
              <p className="text-2xl font-bold text-primary">
                {strengthIndex}/100
              </p>
            </div>
          </motion.div>

          {/* SCHEMES */}
          <motion.div {...fadeUp} className="bg-card rounded-3xl p-8">
            <h2 className="text-xl font-semibold mb-6">
              Recommended Schemes
            </h2>
            <div className="grid sm:grid-cols-2 gap-4">
              {result.eligible_schemes.map((s) => (
                <div
                  key={s.scheme}
                  className="bg-background border rounded-xl p-5"
                >
                  <p className="font-semibold mb-1">{s.scheme}</p>
                  <p className="text-sm text-muted-foreground">
                    {s.reason}
                  </p>
                </div>
              ))}
            </div>
          </motion.div>

          {/* ACTIONS */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => navigate("/eligibility")}
              className="flex items-center gap-2 px-6 py-3 text-sm font-semibold gradient-bg text-primary-foreground rounded-xl"
            >
              <ArrowLeft className="w-4 h-4" /> Recalculate
            </button>

            <button
              onClick={downloadReport}
              className="flex items-center gap-2 px-6 py-3 text-sm font-semibold border rounded-xl"
            >
              <Download className="w-4 h-4" /> Download Report
            </button>
          </div>

        </div>
      </section>
    </PageLayout>
  );
};

export default Results;
