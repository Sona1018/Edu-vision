import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, FileText, CheckCircle2, AlertTriangle, ArrowRight, 
  RotateCcw, Download, Eye, Edit3, Save, X, Search, 
  ChevronDown, ChevronUp, Check, RefreshCw, Layers, FileSpreadsheet,
  AlertCircle, Sparkles, BookOpen, GraduationCap, ShieldCheck
} from 'lucide-react';

const API_BASE = typeof window !== 'undefined' && window.location.port === "3000" ? "http://localhost:8000" : "";

export default function App() {
  // Upload States
  const [studentFile, setStudentFile] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);
  
  const [studentUploading, setStudentUploading] = useState(false);
  const [referenceUploading, setReferenceUploading] = useState(false);
  const [studentUploadProgress, setStudentUploadProgress] = useState(0);
  const [referenceUploadProgress, setReferenceUploadProgress] = useState(0);

  const [studentResult, setStudentResult] = useState(null);
  const [referenceResult, setReferenceResult] = useState(null);

  // Evaluation States
  const [evaluating, setEvaluating] = useState(false);
  const [evalStep, setEvalStep] = useState("");
  const [evaluationData, setEvaluationData] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  // View States
  const [activeTab, setActiveTab] = useState("questions"); // "questions" | "pages"
  const [filterMode, setFilterMode] = useState("all"); // "all" | "review" | "correct"
  const [selectedPageNum, setSelectedPageNum] = useState(1);
  const [expandedQuestions, setExpandedQuestions] = useState({});

  // Manual Override Modal
  const [overrideTarget, setOverrideTarget] = useState(null); // question object
  const [overrideMarks, setOverrideMarks] = useState(0);
  const [overrideRemarks, setOverrideRemarks] = useState("");
  const [overrideAnswer, setOverrideAnswer] = useState("");
  const [savingOverride, setSavingOverride] = useState(false);

  // File Inputs
  const studentInputRef = useRef(null);
  const referenceInputRef = useRef(null);

  // Drag states
  const [dragOverStudent, setDragOverStudent] = useState(false);
  const [dragOverRef, setDragOverRef] = useState(false);

  // 1. Handle Student Upload
  const handleStudentUpload = async (file) => {
    if (!file) return;
    setStudentFile(file);
    setStudentUploading(true);
    setStudentUploadProgress(20);
    setErrorMessage("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setStudentUploadProgress(50);
      const res = await fetch(`${API_BASE}/api/upload/student`, {
        method: "POST",
        body: formData,
      });
      setStudentUploadProgress(90);

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Student sheet processing failed");
      }
      const data = await res.json();
      setStudentResult(data);
      setStudentUploadProgress(100);
    } catch (err) {
      setErrorMessage(err.message);
      setStudentFile(null);
      setStudentResult(null);
    } finally {
      setStudentUploading(false);
    }
  };

  // 2. Handle Reference Upload
  const handleReferenceUpload = async (file) => {
    if (!file) return;
    setReferenceFile(file);
    setReferenceUploading(true);
    setReferenceUploadProgress(20);
    setErrorMessage("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      setReferenceUploadProgress(50);
      const res = await fetch(`${API_BASE}/api/upload/reference`, {
        method: "POST",
        body: formData,
      });
      setReferenceUploadProgress(90);

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Reference key processing failed");
      }
      const data = await res.json();
      setReferenceResult(data);
      setReferenceUploadProgress(100);
    } catch (err) {
      setErrorMessage(err.message);
      setReferenceFile(null);
      setReferenceResult(null);
    } finally {
      setReferenceUploading(false);
    }
  };

  // 3. Central Evaluation Trigger
  const handleEvaluate = async () => {
    if (!studentResult || !referenceResult) return;
    setEvaluating(true);
    setErrorMessage("");
    setEvalStep("Initializing SBERT Semantic Engine & DeBERTa NLI...");

    try {
      // Build request payload
      const payload = {
        student_file_id: studentResult.file_id,
        reference_file_id: referenceResult.file_id,
        extracted_answers: studentResult.extracted_answers || studentResult.questions_parsed || {},
        qa_bank: referenceResult.questions || referenceResult.qa_bank || {}
      };

      setEvalStep("Evaluating question-by-question semantics, rubrics, and structural depth...");
      const res = await fetch(`${API_BASE}/api/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Evaluation failed");
      }

      const evalData = await res.json();
      setEvaluationData(evalData);
      // Auto expand all questions initially
      const initialExpanded = {};
      evalData.questions.forEach(q => {
        initialExpanded[q.question_id] = true;
      });
      setExpandedQuestions(initialExpanded);
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setEvaluating(false);
      setEvalStep("");
    }
  };

  // 4. Save Manual Override
  const handleSaveOverride = async () => {
    if (!overrideTarget || !evaluationData) return;
    setSavingOverride(true);
    try {
      const qNum = overrideTarget.question_number || parseInt(overrideTarget.question_id.replace("Q", "")) || 1;
      const res = await fetch(`${API_BASE}/api/results/${evaluationData.evaluation_id}/override`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question_number: qNum,
          awarded_marks: parseFloat(overrideMarks),
          remarks: overrideRemarks,
          updated_answer: overrideAnswer
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to save override");
      }

      const updated = await res.json();
      setEvaluationData(updated);
      setOverrideTarget(null);
    } catch (err) {
      alert("Override failed: " + err.message);
    } finally {
      setSavingOverride(false);
    }
  };

  // 5. Download Exports
  const handleDownloadPDF = () => {
    if (!evaluationData) return;
    window.open(`${API_BASE}/api/export/${evaluationData.evaluation_id}/pdf`, "_blank");
  };

  const handleDownloadExcel = () => {
    if (!evaluationData) return;
    window.open(`${API_BASE}/api/export/${evaluationData.evaluation_id}/excel`, "_blank");
  };

  // Reset all
  const handleReset = () => {
    setStudentFile(null);
    setReferenceFile(null);
    setStudentResult(null);
    setReferenceResult(null);
    setEvaluationData(null);
    setErrorMessage("");
    setActiveTab("questions");
  };

  // Filter questions
  const filteredQuestions = (evaluationData?.questions || []).filter(q => {
    if (filterMode === "review") return q.needs_manual_review;
    if (filterMode === "correct") return q.status_badge === "correct";
    return true;
  });

  return (
    <div className="min-h-screen bg-[#09090b] text-[#f4f4f5] font-sans flex flex-col antialiased">
      {/* Top Navbar */}
      <header className="border-b border-[#27272a] bg-[#09090b]/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded border border-neutral-700 bg-neutral-900 flex items-center justify-center font-bold text-white tracking-wider">
              IP
            </div>
            <div>
              <span className="font-semibold text-white tracking-tight text-base">IntelliPlace</span>
              <span className="text-neutral-500 text-xs ml-2 border border-neutral-800 px-1.5 py-0.5 rounded font-mono">v2.0</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-xs text-neutral-400">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>FastAPI Backend Active</span>
            </div>
            {evaluationData && (
              <button 
                onClick={handleReset}
                className="flex items-center space-x-1.5 text-xs border border-neutral-800 hover:border-neutral-600 bg-neutral-950 px-3 py-1.5 rounded text-neutral-300 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>New Evaluation</span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-6 py-8 flex-1 w-full">
        {/* Error Notification */}
        {errorMessage && (
          <div className="mb-6 p-4 rounded border border-neutral-700 bg-neutral-900/90 text-neutral-200 flex items-start justify-between text-sm">
            <div className="flex items-center space-x-3">
              <AlertTriangle className="w-5 h-5 text-neutral-400 flex-shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button onClick={() => setErrorMessage("")} className="text-neutral-500 hover:text-white">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* View 1: Upload & Pipeline Trigger Screen */}
        {!evaluationData && (
          <div className="space-y-10 py-4">
            {/* Hero Title */}
            <div className="text-center max-w-2xl mx-auto space-y-3">
              <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white">
                Answer Evaluation System
              </h1>
              <p className="text-neutral-400 text-sm sm:text-base leading-relaxed">
                Automated handwritten answer sheet evaluation powered by OCR and intelligent assessment.
              </p>
            </div>

            {/* TWO SIDE-BY-SIDE UPLOAD BOXES */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-5xl mx-auto">
              {/* BOX 1: Student Answer Sheet */}
              <div 
                onDragOver={(e) => { e.preventDefault(); setDragOverStudent(true); }}
                onDragLeave={() => setDragOverStudent(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOverStudent(false);
                  if (e.dataTransfer.files?.[0]) handleStudentUpload(e.dataTransfer.files[0]);
                }}
                className={`border rounded-lg p-6 bg-[#121214] flex flex-col justify-between transition-all ${
                  dragOverStudent ? 'border-white bg-neutral-900' : 'border-[#27272a] hover:border-neutral-700'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-mono tracking-wider text-neutral-400 uppercase">PIPELINE 1 : OCR MODEL</span>
                    {studentResult && (
                      <span className="text-xs px-2 py-0.5 rounded bg-neutral-800 text-neutral-200 border border-neutral-700 flex items-center space-x-1">
                        <Check className="w-3 h-3 text-white" />
                        <span>Ready</span>
                      </span>
                    )}
                  </div>

                  <h3 className="text-lg font-semibold text-white mb-1">Upload Student Answer Sheet</h3>
                  <p className="text-neutral-400 text-xs mb-5">
                    Upload the handwritten answer sheet that needs to be evaluated.
                  </p>

                  <input 
                    type="file" 
                    ref={studentInputRef} 
                    onChange={(e) => handleStudentUpload(e.target.files?.[0])}
                    accept=".pdf,.png,.jpg,.jpeg" 
                    className="hidden" 
                  />

                  {/* Drag drop area */}
                  {!studentFile ? (
                    <div 
                      onClick={() => studentInputRef.current?.click()}
                      className="border border-dashed border-neutral-700 hover:border-neutral-500 rounded-lg p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-colors bg-[#09090b]/50"
                    >
                      <div className="w-10 h-10 rounded-full border border-neutral-800 bg-neutral-900 flex items-center justify-center mb-3">
                        <Upload className="w-5 h-5 text-neutral-400" />
                      </div>
                      <p className="text-sm font-medium text-white mb-1">Click to upload or drag & drop</p>
                      <p className="text-xs text-neutral-500">Supports PDF, PNG, JPG (Multi-page supported)</p>
                    </div>
                  ) : (
                    <div className="border border-neutral-800 rounded-lg p-4 bg-neutral-950 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3 overflow-hidden">
                          <FileText className="w-6 h-6 text-neutral-400 flex-shrink-0" />
                          <div className="truncate">
                            <p className="text-sm font-medium text-white truncate">{studentFile.name}</p>
                            <p className="text-xs text-neutral-500">{(studentFile.size / 1024).toFixed(1)} KB</p>
                          </div>
                        </div>
                        <button 
                          onClick={() => { setStudentFile(null); setStudentResult(null); }}
                          className="text-neutral-500 hover:text-white p-1"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>

                      {studentUploading && (
                        <div className="space-y-1">
                          <div className="w-full bg-neutral-800 h-1 rounded overflow-hidden">
                            <div className="bg-white h-full transition-all duration-300" style={{ width: `${studentUploadProgress}%` }}></div>
                          </div>
                          <p className="text-[11px] text-neutral-400 font-mono">Running Tesseract & PyMuPDF OCR...</p>
                        </div>
                      )}

                      {studentResult && (
                        <div className="pt-2 border-t border-neutral-900 text-xs text-neutral-400 space-y-1 font-mono">
                          <div className="flex justify-between">
                            <span>Pages Rendered:</span>
                            <span className="text-white">{studentResult.total_pages || studentResult.pages?.length || 1}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Questions Detected:</span>
                            <span className="text-white">
                              {Object.keys(studentResult.extracted_answers || studentResult.questions_parsed || {}).length}
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div className="mt-5 pt-4 border-t border-[#27272a]/60 flex items-center justify-between">
                  <span className="text-[11px] text-neutral-500">Isolated OCR Pipeline</span>
                  <button 
                    onClick={() => studentInputRef.current?.click()}
                    disabled={studentUploading}
                    className="text-xs border border-neutral-700 hover:border-white px-3 py-1.5 rounded text-neutral-200 transition-colors bg-neutral-900"
                  >
                    {studentFile ? "Replace File" : "Choose File"}
                  </button>
                </div>
              </div>

              {/* BOX 2: Reference Answer Key */}
              <div 
                onDragOver={(e) => { e.preventDefault(); setDragOverRef(true); }}
                onDragLeave={() => setDragOverRef(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOverRef(false);
                  if (e.dataTransfer.files?.[0]) handleReferenceUpload(e.dataTransfer.files[0]);
                }}
                className={`border rounded-lg p-6 bg-[#121214] flex flex-col justify-between transition-all ${
                  dragOverRef ? 'border-white bg-neutral-900' : 'border-[#27272a] hover:border-neutral-700'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-mono tracking-wider text-neutral-400 uppercase">PIPELINE 2 : REFERENCE MODEL</span>
                    {referenceResult && (
                      <span className="text-xs px-2 py-0.5 rounded bg-neutral-800 text-neutral-200 border border-neutral-700 flex items-center space-x-1">
                        <Check className="w-3 h-3 text-white" />
                        <span>Ready</span>
                      </span>
                    )}
                  </div>

                  <h3 className="text-lg font-semibold text-white mb-1">Upload Reference Answer Key</h3>
                  <p className="text-neutral-400 text-xs mb-5">
                    Upload the official reference answers and marking scheme.
                  </p>

                  <input 
                    type="file" 
                    ref={referenceInputRef} 
                    onChange={(e) => handleReferenceUpload(e.target.files?.[0])}
                    accept=".txt,.pdf,.docx" 
                    className="hidden" 
                  />

                  {/* Drag drop area */}
                  {!referenceFile ? (
                    <div 
                      onClick={() => referenceInputRef.current?.click()}
                      className="border border-dashed border-neutral-700 hover:border-neutral-500 rounded-lg p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-colors bg-[#09090b]/50"
                    >
                      <div className="w-10 h-10 rounded-full border border-neutral-800 bg-neutral-900 flex items-center justify-center mb-3">
                        <FileText className="w-5 h-5 text-neutral-400" />
                      </div>
                      <p className="text-sm font-medium text-white mb-1">Click to upload or drag & drop</p>
                      <p className="text-xs text-neutral-500">Supports TXT, PDF, DOCX (Auto-extracts rubrics)</p>
                    </div>
                  ) : (
                    <div className="border border-neutral-800 rounded-lg p-4 bg-neutral-950 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3 overflow-hidden">
                          <FileText className="w-6 h-6 text-neutral-400 flex-shrink-0" />
                          <div className="truncate">
                            <p className="text-sm font-medium text-white truncate">{referenceFile.name}</p>
                            <p className="text-xs text-neutral-500">{(referenceFile.size / 1024).toFixed(1)} KB</p>
                          </div>
                        </div>
                        <button 
                          onClick={() => { setReferenceFile(null); setReferenceResult(null); }}
                          className="text-neutral-500 hover:text-white p-1"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>

                      {referenceUploading && (
                        <div className="space-y-1">
                          <div className="w-full bg-neutral-800 h-1 rounded overflow-hidden">
                            <div className="bg-white h-full transition-all duration-300" style={{ width: `${referenceUploadProgress}%` }}></div>
                          </div>
                          <p className="text-[11px] text-neutral-400 font-mono">Parsing QA Bank & Normalized Rubrics...</p>
                        </div>
                      )}

                      {referenceResult && (
                        <div className="pt-2 border-t border-neutral-900 text-xs text-neutral-400 space-y-1 font-mono">
                          <div className="flex justify-between">
                            <span>Reference Questions:</span>
                            <span className="text-white">{referenceResult.total_questions || referenceResult.questions?.length || 0}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Total Max Marks:</span>
                            <span className="text-white">{referenceResult.total_max_marks || 0}</span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div className="mt-5 pt-4 border-t border-[#27272a]/60 flex items-center justify-between">
                  <span className="text-[11px] text-neutral-500">Separate Evaluation Model</span>
                  <button 
                    onClick={() => referenceInputRef.current?.click()}
                    disabled={referenceUploading}
                    className="text-xs border border-neutral-700 hover:border-white px-3 py-1.5 rounded text-neutral-200 transition-colors bg-neutral-900"
                  >
                    {referenceFile ? "Replace File" : "Choose File"}
                  </button>
                </div>
              </div>
            </div>

            {/* Central Evaluation Button */}
            <div className="max-w-md mx-auto text-center space-y-3 pt-4">
              <button
                onClick={handleEvaluate}
                disabled={!studentResult || !referenceResult || evaluating}
                className={`w-full py-3.5 px-6 rounded-lg font-semibold text-sm transition-all duration-200 flex items-center justify-center space-x-2 ${
                  studentResult && referenceResult && !evaluating
                    ? 'bg-white text-black hover:bg-neutral-200 shadow-lg cursor-pointer active:scale-[0.99]'
                    : 'bg-neutral-900 text-neutral-600 border border-neutral-800 cursor-not-allowed'
                }`}
              >
                {evaluating ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-neutral-400" />
                    <span className="text-neutral-300">Evaluating Answer Sheet...</span>
                  </>
                ) : (
                  <>
                    <span>Evaluate Answer Sheet</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>

              {evaluating && (
                <div className="space-y-2">
                  <div className="w-full bg-neutral-900 border border-neutral-800 rounded-full h-1.5 overflow-hidden">
                    <div className="bg-white h-full animate-pulse w-3/4"></div>
                  </div>
                  <p className="text-xs text-neutral-400 font-mono">{evalStep}</p>
                </div>
              )}

              {!studentResult || !referenceResult ? (
                <p className="text-xs text-neutral-500">
                  Upload both the Student Answer Sheet and Reference Answer Key to start evaluation.
                </p>
              ) : null}
            </div>
          </div>
        )}

        {/* View 2: Results Dashboard */}
        {evaluationData && (
          <div className="space-y-8">
            {/* Action Bar & Scorecard Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-[#27272a]">
              <div>
                <span className="text-xs font-mono text-neutral-500 uppercase tracking-wider">EVALUATION REPORT</span>
                <h2 className="text-2xl font-bold text-white tracking-tight">Assessment Overview</h2>
              </div>
              <div className="flex items-center space-x-3">
                <button
                  onClick={handleDownloadPDF}
                  className="flex items-center space-x-2 text-xs border border-neutral-700 hover:border-white px-3.5 py-2 rounded bg-neutral-900 text-neutral-200 transition-colors"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download PDF</span>
                </button>
                <button
                  onClick={handleDownloadExcel}
                  className="flex items-center space-x-2 text-xs border border-neutral-700 hover:border-white px-3.5 py-2 rounded bg-neutral-900 text-neutral-200 transition-colors"
                >
                  <FileSpreadsheet className="w-3.5 h-3.5" />
                  <span>Export Excel</span>
                </button>
              </div>
            </div>

            {/* Overall Performance Metric Cards */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="p-4 rounded-lg bg-[#121214] border border-[#27272a] space-y-1">
                <p className="text-xs font-mono text-neutral-500">TOTAL SCORE</p>
                <p className="text-2xl font-bold text-white">
                  {evaluationData.total_marks_obtained} <span className="text-xs text-neutral-500 font-normal">/ {evaluationData.total_max_marks}</span>
                </p>
              </div>
              <div className="p-4 rounded-lg bg-[#121214] border border-[#27272a] space-y-1">
                <p className="text-xs font-mono text-neutral-500">PERCENTAGE</p>
                <p className="text-2xl font-bold text-white">{evaluationData.overall_percentage}%</p>
              </div>
              <div className="p-4 rounded-lg bg-[#121214] border border-[#27272a] space-y-1">
                <p className="text-xs font-mono text-neutral-500">GRADE</p>
                <p className="text-2xl font-bold text-white">{evaluationData.overall_grade}</p>
              </div>
              <div className="p-4 rounded-lg bg-[#121214] border border-[#27272a] space-y-1">
                <p className="text-xs font-mono text-neutral-500">QUESTIONS</p>
                <p className="text-2xl font-bold text-white">{evaluationData.total_questions_evaluated}</p>
              </div>
              <div className="p-4 rounded-lg bg-[#121214] border border-[#27272a] space-y-1 col-span-2 md:col-span-1">
                <p className="text-xs font-mono text-neutral-500">MANUAL REVIEW</p>
                <p className={`text-2xl font-bold ${evaluationData.questions_requiring_manual_review > 0 ? 'text-white' : 'text-neutral-400'}`}>
                  {evaluationData.questions_requiring_manual_review} Flags
                </p>
              </div>
            </div>

            {/* Navigation Tabs (Question-wise vs Page-wise) */}
            <div className="flex items-center justify-between border-b border-[#27272a] pb-2">
              <div className="flex space-x-2">
                <button
                  onClick={() => setActiveTab("questions")}
                  className={`px-4 py-2 text-xs font-medium rounded transition-colors ${
                    activeTab === "questions" 
                      ? 'bg-white text-black font-semibold' 
                      : 'text-neutral-400 hover:text-white hover:bg-neutral-900'
                  }`}
                >
                  Question-wise Evaluation ({evaluationData.questions.length})
                </button>
                <button
                  onClick={() => setActiveTab("pages")}
                  className={`px-4 py-2 text-xs font-medium rounded transition-colors ${
                    activeTab === "pages" 
                      ? 'bg-white text-black font-semibold' 
                      : 'text-neutral-400 hover:text-white hover:bg-neutral-900'
                  }`}
                >
                  Page-wise Handwriting vs OCR ({evaluationData.pages?.length || 1} Pages)
                </button>
              </div>

              {activeTab === "questions" && (
                <div className="flex items-center space-x-2 text-xs">
                  <span className="text-neutral-500">Filter:</span>
                  <button 
                    onClick={() => setFilterMode("all")}
                    className={`px-2.5 py-1 rounded border ${filterMode === "all" ? 'border-white text-white' : 'border-neutral-800 text-neutral-500'}`}
                  >
                    All
                  </button>
                  <button 
                    onClick={() => setFilterMode("review")}
                    className={`px-2.5 py-1 rounded border ${filterMode === "review" ? 'border-white text-white' : 'border-neutral-800 text-neutral-500'}`}
                  >
                    Flagged ({evaluationData.questions.filter(q => q.needs_manual_review).length})
                  </button>
                  <button 
                    onClick={() => setFilterMode("correct")}
                    className={`px-2.5 py-1 rounded border ${filterMode === "correct" ? 'border-white text-white' : 'border-neutral-800 text-neutral-500'}`}
                  >
                    High Marks
                  </button>
                </div>
              )}
            </div>

            {/* TAB 1: QUESTION-WISE EVALUATION */}
            {activeTab === "questions" && (
              <div className="space-y-4">
                {filteredQuestions.map((q) => {
                  const isExpanded = expandedQuestions[q.question_id] !== false;
                  return (
                    <div 
                      key={q.question_id}
                      className="border border-[#27272a] rounded-lg bg-[#121214] overflow-hidden transition-all"
                    >
                      {/* Header bar of question card */}
                      <div 
                        onClick={() => setExpandedQuestions(prev => ({ ...prev, [q.question_id]: !isExpanded }))}
                        className="p-4 flex items-center justify-between cursor-pointer hover:bg-neutral-900/60 select-none border-b border-transparent"
                      >
                        <div className="flex items-center space-x-3">
                          <span className="w-7 h-7 rounded border border-neutral-700 bg-neutral-950 flex items-center justify-center font-mono text-xs font-bold text-white">
                            {q.question_id}
                          </span>
                          <div>
                            <span className="text-sm font-semibold text-white mr-2">
                              {q.question_text || `Question ${q.question_number || q.question_id}`}
                            </span>
                            {q.needs_manual_review && (
                              <span className="text-[10px] px-2 py-0.5 rounded border border-neutral-600 bg-neutral-900 text-neutral-300 font-mono uppercase">
                                Needs Manual Review
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="flex items-center space-x-4">
                          <div className="text-right">
                            <span className="text-base font-bold text-white">{q.marks_awarded}</span>
                            <span className="text-xs text-neutral-500"> / {q.max_marks} Marks</span>
                          </div>
                          {isExpanded ? <ChevronUp className="w-4 h-4 text-neutral-500" /> : <ChevronDown className="w-4 h-4 text-neutral-500" />}
                        </div>
                      </div>

                      {/* Expandable details */}
                      {isExpanded && (
                        <div className="p-5 border-t border-[#27272a] space-y-5 bg-[#0e0e10]/40">
                          {/* Student Answer & Reference Answer Grid */}
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="p-4 rounded border border-neutral-800 bg-[#121214] space-y-2">
                              <div className="flex items-center justify-between">
                                <span className="text-xs font-mono text-neutral-400 uppercase">EXTRACTED STUDENT ANSWER</span>
                                <button 
                                  onClick={() => {
                                    setOverrideTarget(q);
                                    setOverrideMarks(q.marks_awarded);
                                    setOverrideRemarks(q.teacher_remarks || "");
                                    setOverrideAnswer(q.extracted_student_answer || "");
                                  }}
                                  className="text-[11px] text-neutral-400 hover:text-white flex items-center space-x-1"
                                >
                                  <Edit3 className="w-3 h-3" />
                                  <span>Edit / Override</span>
                                </button>
                              </div>
                              <p className="text-xs text-neutral-300 leading-relaxed font-mono whitespace-pre-wrap bg-neutral-950 p-3 rounded border border-neutral-900 max-h-48 overflow-y-auto">
                                {q.extracted_student_answer || "No answer text detected for this question."}
                              </p>
                            </div>

                            <div className="p-4 rounded border border-neutral-800 bg-[#121214] space-y-2">
                              <span className="text-xs font-mono text-neutral-400 uppercase">REFERENCE ANSWER / RUBRIC</span>
                              <p className="text-xs text-neutral-400 leading-relaxed font-mono whitespace-pre-wrap bg-neutral-950 p-3 rounded border border-neutral-900 max-h-48 overflow-y-auto">
                                {q.reference_answer || "Reference answer specification."}
                              </p>
                            </div>
                          </div>

                          {/* AI Rubric & Evaluation Breakdown */}
                          <div className="p-4 rounded border border-neutral-800 bg-neutral-950/70 space-y-3">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-2">
                                <Sparkles className="w-4 h-4 text-neutral-300" />
                                <span className="text-xs font-mono uppercase text-white font-semibold">Evaluation Breakdown & Remarks</span>
                              </div>
                              <span className="text-xs font-mono text-neutral-500">Verdict: {q.verdict}</span>
                            </div>
                            
                            <p className="text-xs text-neutral-300 leading-relaxed">
                              {q.explanation}
                            </p>

                            {/* Missing Concepts */}
                            {q.missing_key_concepts && q.missing_key_concepts.length > 0 && (
                              <div className="pt-2 border-t border-neutral-900">
                                <p className="text-[11px] font-mono text-neutral-400 mb-1.5 uppercase">Missing Key Concepts:</p>
                                <div className="flex flex-wrap gap-1.5">
                                  {q.missing_key_concepts.map((concept, idx) => (
                                    <span key={idx} className="text-[11px] px-2 py-0.5 rounded border border-neutral-700 bg-neutral-900 text-neutral-300 font-mono">
                                      {concept}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Incorrect statements */}
                            {q.incorrect_statements && q.incorrect_statements.length > 0 && (
                              <div className="pt-2 border-t border-neutral-900">
                                <p className="text-[11px] font-mono text-neutral-400 mb-1.5 uppercase">Contradictory / Incorrect Points:</p>
                                <ul className="text-xs text-neutral-400 list-disc list-inside space-y-1">
                                  {q.incorrect_statements.map((s, idx) => (
                                    <li key={idx}>{s}</li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Suggestions */}
                            {q.suggestions_for_improvement && q.suggestions_for_improvement.length > 0 && (
                              <div className="pt-2 border-t border-neutral-900">
                                <p className="text-[11px] font-mono text-neutral-400 mb-1.5 uppercase">Suggestions For Improvement:</p>
                                <ul className="text-xs text-neutral-400 list-disc list-inside space-y-1">
                                  {q.suggestions_for_improvement.map((sug, idx) => (
                                    <li key={idx}>{sug}</li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Teacher Remarks if overridden */}
                            {q.teacher_remarks && (
                              <div className="pt-2 border-t border-neutral-800 text-xs">
                                <span className="font-mono text-white font-semibold">Evaluator Notes: </span>
                                <span className="text-neutral-300 italic">{q.teacher_remarks}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* TAB 2: PAGE-WISE HANDWRITING VS OCR COMPARISON */}
            {activeTab === "pages" && (
              <div className="space-y-6">
                {/* Page selector buttons */}
                <div className="flex items-center space-x-2 border-b border-[#27272a] pb-3 overflow-x-auto">
                  <span className="text-xs text-neutral-500 font-mono">SELECT PAGE:</span>
                  {(evaluationData.pages || [{ page_number: 1 }]).map((p, idx) => {
                    const pNum = p.page_number || idx + 1;
                    return (
                      <button
                        key={pNum}
                        onClick={() => setSelectedPageNum(pNum)}
                        className={`px-3 py-1 text-xs font-mono rounded border transition-colors ${
                          selectedPageNum === pNum 
                            ? 'border-white bg-white text-black font-semibold' 
                            : 'border-neutral-800 text-neutral-400 hover:border-neutral-600'
                        }`}
                      >
                        Page {pNum}
                      </button>
                    );
                  })}
                </div>

                {/* Side-by-side Page Comparison */}
                {(() => {
                  const currPage = (evaluationData.pages || []).find(p => (p.page_number || 1) === selectedPageNum) || (evaluationData.pages?.[0] || {});
                  const imageUrl = currPage.image_url ? `${API_BASE}${currPage.image_url}` : null;
                  
                  return (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* Left Side: Original Handwriting Page Image */}
                      <div className="border border-[#27272a] rounded-lg p-4 bg-[#121214] space-y-3">
                        <div className="flex items-center justify-between pb-2 border-b border-neutral-800">
                          <span className="text-xs font-mono text-neutral-400 uppercase">ORIGINAL STUDENT HANDWRITING (PAGE {selectedPageNum})</span>
                          <span className="text-[11px] font-mono text-neutral-500">High-Resolution Render</span>
                        </div>
                        <div className="bg-neutral-950 rounded border border-neutral-800 flex items-center justify-center p-2 min-h-[500px] overflow-hidden">
                          {imageUrl ? (
                            <img 
                              src={imageUrl} 
                              alt={`Student Sheet Page ${selectedPageNum}`} 
                              className="max-h-[650px] w-auto object-contain rounded shadow"
                            />
                          ) : (
                            <div className="text-center text-neutral-600 text-xs font-mono">
                              Page image rendered in backend storage.
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Right Side: Extracted OCR Text & Page Questions */}
                      <div className="border border-[#27272a] rounded-lg p-4 bg-[#121214] flex flex-col justify-between space-y-4">
                        <div className="space-y-4">
                          <div className="flex items-center justify-between pb-2 border-b border-neutral-800">
                            <span className="text-xs font-mono text-neutral-400 uppercase">EXTRACTED OCR TEXT & QUESTIONS</span>
                            <span className="text-[11px] font-mono text-neutral-500">Pipeline 1 Output</span>
                          </div>

                          {/* Detected Questions on this page */}
                          <div>
                            <p className="text-[11px] font-mono text-neutral-500 uppercase mb-2">Detected Questions On This Page:</p>
                            <div className="flex flex-wrap gap-2">
                              {(currPage.detected_questions || []).length > 0 ? (
                                currPage.detected_questions.map(qId => (
                                  <span key={qId} className="text-xs font-mono px-2.5 py-1 rounded border border-neutral-700 bg-neutral-900 text-white">
                                    Question {qId}
                                  </span>
                                ))
                              ) : (
                                <span className="text-xs text-neutral-500 font-mono">Continuous answer flow</span>
                              )}
                            </div>
                          </div>

                          {/* Full extracted page text */}
                          <div>
                            <p className="text-[11px] font-mono text-neutral-500 uppercase mb-2">Raw OCR Text Stream:</p>
                            <div className="bg-neutral-950 p-4 rounded border border-neutral-800 text-xs font-mono text-neutral-300 whitespace-pre-wrap leading-relaxed max-h-[500px] overflow-y-auto">
                              {currPage.text || currPage.raw_text || "No text extracted for this page."}
                            </div>
                          </div>
                        </div>

                        <div className="pt-3 border-t border-neutral-800 flex items-center justify-between text-xs text-neutral-500">
                          <span>Verify handwriting accuracy before grading</span>
                          <button 
                            onClick={() => setActiveTab("questions")}
                            className="text-white hover:underline flex items-center space-x-1"
                          >
                            <span>Return to Question Grading</span>
                            <ArrowRight className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        )}

        {/* Modal: Manual Override / Teacher Grading Correction */}
        {overrideTarget && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-[#121214] border border-neutral-700 rounded-lg max-w-lg w-full p-6 space-y-5 shadow-2xl">
              <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
                <div>
                  <span className="text-xs font-mono text-neutral-500 uppercase">MANUAL OVERRIDE & AUDIT</span>
                  <h3 className="text-lg font-bold text-white">Question {overrideTarget.question_number || overrideTarget.question_id}</h3>
                </div>
                <button onClick={() => setOverrideTarget(null)} className="text-neutral-500 hover:text-white">
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-4">
                {/* Marks input */}
                <div>
                  <label className="block text-xs font-mono text-neutral-400 mb-1 uppercase">
                    Awarded Marks (Max: {overrideTarget.max_marks})
                  </label>
                  <input 
                    type="number" 
                    step="0.5"
                    min="0"
                    max={overrideTarget.max_marks}
                    value={overrideMarks}
                    onChange={(e) => setOverrideMarks(e.target.value)}
                    className="w-full bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-sm text-white font-mono focus:border-white outline-none"
                  />
                </div>

                {/* Edit student answer text */}
                <div>
                  <label className="block text-xs font-mono text-neutral-400 mb-1 uppercase">
                    Correct / Edit Student Answer Text
                  </label>
                  <textarea 
                    rows={4}
                    value={overrideAnswer}
                    onChange={(e) => setOverrideAnswer(e.target.value)}
                    className="w-full bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-xs text-neutral-200 font-mono focus:border-white outline-none leading-relaxed"
                  />
                </div>

                {/* Remarks input */}
                <div>
                  <label className="block text-xs font-mono text-neutral-400 mb-1 uppercase">
                    Evaluator Remarks / Justification
                  </label>
                  <textarea 
                    rows={2}
                    value={overrideRemarks}
                    onChange={(e) => setOverrideRemarks(e.target.value)}
                    placeholder="Enter reason for mark adjustment or handwriting correction..."
                    className="w-full bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-xs text-neutral-200 font-mono focus:border-white outline-none"
                  />
                </div>
              </div>

              <div className="flex items-center justify-end space-x-3 pt-3 border-t border-neutral-800">
                <button 
                  onClick={() => setOverrideTarget(null)}
                  className="px-4 py-2 rounded text-xs border border-neutral-800 text-neutral-400 hover:text-white"
                >
                  Cancel
                </button>
                <button 
                  onClick={handleSaveOverride}
                  disabled={savingOverride}
                  className="px-4 py-2 rounded text-xs font-semibold bg-white text-black hover:bg-neutral-200 transition-colors flex items-center space-x-1.5"
                >
                  {savingOverride ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>Saving...</span>
                    </>
                  ) : (
                    <>
                      <Save className="w-3.5 h-3.5" />
                      <span>Save & Recalculate Total</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Monochromatic Footer */}
      <footer className="border-t border-[#27272a] py-6 text-center text-xs text-neutral-500 font-mono">
        <p>IntelliPlace AI Answer Sheet Evaluation System &bull; Pure Monochromatic Clean Architecture</p>
      </footer>
    </div>
  );
}
