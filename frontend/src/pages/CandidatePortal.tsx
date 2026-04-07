import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Briefcase, Settings, ChevronRight, User, CheckCircle2,
  Loader2, Save, X, UploadCloud, FileText, Keyboard, Sparkles
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const API = 'http://127.0.0.1:8000/api';

export default function CandidatePortal() {
  const [profile, setProfile] = useState<any>(null);
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [selectedJob, setSelectedJob] = useState<any>(null);
  const [interactionMessage, setInteractionMessage] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [entryMode, setEntryMode] = useState<'upload' | 'manual'>('upload');
  const [setupComplete, setSetupComplete] = useState(false);
  const [appliedJobs, setAppliedJobs] = useState<Set<string>>(new Set());

  // Resume upload state
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [parsing, setParsing] = useState(false);
  const [parsedData, setParsedData] = useState<any>(null);

  const [editData, setEditData] = useState({
    skills: '', salary_current: '', experience_level: 'mid', contract_preference: 'permanent',
    description: ''
  });

  const navigate = useNavigate();

  const fetchProfile = useCallback(async () => {
    const token = localStorage.getItem('token');
    if (!token) return navigate('/auth');
    try {
      const res = await fetch(`${API}/auth/me/`, {
        headers: { Authorization: `Token ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        if (data.role !== 'candidate') return navigate('/auth');
        setProfile(data.profile);
        if (data.profile) {
          setSetupComplete(true);
          setEditData({
            skills: (data.profile.skills || []).join(', '),
            salary_current: data.profile.salary_current ?? '',
            experience_level: data.profile.experience_bucket || 'mid',
            contract_preference: data.profile.contract_preference || 'permanent',
            description: data.profile.description || ''
          });
          setLoadingRecommendations(true);
          const recRes = await fetch(`${API}/recommend/jobs/`, {
            headers: { Authorization: `Token ${token}` }
          });
          if (recRes.ok) {
            const recData = await recRes.json();
            setRecommendations(recData.recommendations || []);
          }
          setLoadingRecommendations(false);
        }
      } else {
        localStorage.removeItem('token');
        navigate('/auth');
      }
    } catch (err) {
      console.error('Fetch failed:', err);
    }
    setLoading(false);
  }, [navigate]);

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) setFile(dropped);
  }, []);

  const handleParseResume = async () => {
    if (!file) return;
    setParsing(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API}/parse-resume/`, { method: 'POST', body: fd });
      const data = await res.json();
      if (res.ok) {
        setParsedData(data.data);
        setEditData({
          skills: (data.data.skills || []).join(', '),
          salary_current: data.data.salary_current ?? '',
          experience_level: data.data.experience_level || data.data.experience_bucket || 'mid',
          contract_preference: data.data.contract_preference || 'permanent',
          description: data.data.description || ''
        });
      }
    } catch (err) { console.error(err); }
    setParsing(false);
  };

  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    const token = localStorage.getItem('token');
    try {
      const res = await fetch(`${API}/auth/update/`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Token ${token}` },
        body: JSON.stringify({
          skills: editData.skills.split(',').map(s => s.trim()).filter(Boolean),
          salary_current: editData.salary_current ? Number(editData.salary_current) : null,
          experience_level: editData.experience_level,
          contract_preference: editData.contract_preference,
          description: editData.description
        })
      });
      if (res.ok) {
        await fetchProfile();
        setEditing(false);
        setSetupComplete(true);
      }
    } catch (err) { console.error(err); }
    setSaving(false);
  };

  const handleApply = async (jobId: string) => {
    if (!jobId) return;
    const token = localStorage.getItem('token');
    if (!token) return;

    try {
      const res = await fetch(`${API}/log-interaction/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Token ${token}` },
        body: JSON.stringify({ job_id: jobId, action: 'application' })
      });
      if (!res.ok) {
        throw new Error(`Interaction failed with status ${res.status}`);
      }

      const payload = await res.json();
      setAppliedJobs(prev => new Set(prev).add(jobId));
      if (payload?.retrain_due) {
        setInteractionMessage('Application logged. Retraining threshold reached, refresh recommendations shortly.');
      } else {
        setInteractionMessage('Application logged successfully.');
      }
    } catch (err) { console.error('Failed to log interaction', err); }
  };

  const inputCls = 'w-full bg-white/5 border border-white/10 rounded-lg py-2 px-3 text-white text-sm focus:outline-none focus:ring-1 focus:ring-white transition-all';
  const labelCls = 'text-[10px] uppercase tracking-widest text-gray-500 font-bold block mb-1';

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-80px)] flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-white/40" />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-10">
        <h1 className="text-3xl font-bold tracking-tight text-white">
          Candidate <span className="text-gray-500 font-mono text-xl">Workspace</span>
        </h1>
      </div>

      <div className="grid lg:grid-cols-3 gap-8 items-start">
        {/* === LEFT: Profile / Setup Panel === */}
        <div className="lg:col-span-1 border border-white/10 bg-[#050505] rounded-3xl p-6 relative">
          <div className="absolute top-4 right-4">
            {setupComplete && (
              <button
                onClick={() => setEditing(!editing)}
                className="p-2 bg-white/5 hover:bg-white/10 rounded-xl transition-colors border border-white/5"
              >
                {editing ? <X className="w-4 h-4 text-white" /> : <Settings className="w-4 h-4 text-gray-400" />}
              </button>
            )}
          </div>

          {profile && (
            <div className="flex items-center mb-6">
              <div className="w-11 h-11 bg-white/5 rounded-full flex items-center justify-center border border-white/10">
                <User className="w-5 h-5 text-white" />
              </div>
              <div className="ml-3">
                <h2 className="text-base font-bold text-white">{profile.name}</h2>
                <p className="text-[10px] font-mono text-gray-500 uppercase tracking-widest">{profile.candidate_ID}</p>
              </div>
            </div>
          )}

          <AnimatePresence mode="wait">
            {!setupComplete ? (
              /* === SETUP: Upload OR Manual === */
              <motion.div key="setup" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                {/* Segmented Control */}
                <div className="relative flex p-1 bg-black/50 rounded-2xl mb-6 border border-white/5">
                  {(['upload', 'manual'] as const).map(mode => (
                    <button
                      key={mode}
                      onClick={() => setEntryMode(mode)}
                      className={`relative flex-1 py-2.5 text-xs font-bold uppercase tracking-widest rounded-xl z-10 flex items-center justify-center gap-2 transition-colors ${entryMode === mode ? 'text-white' : 'text-gray-500 hover:text-gray-300'}`}
                    >
                      {entryMode === mode && (
                        <motion.div layoutId="tab-indicator" className="absolute inset-0 bg-white/10 border border-white/10 rounded-xl" transition={{ type: 'spring', bounce: 0.2, duration: 0.5 }} />
                      )}
                      <span className="relative z-10 flex items-center gap-2">
                        {mode === 'upload' ? <FileText className="w-3.5 h-3.5" /> : <Keyboard className="w-3.5 h-3.5" />}
                        {mode}
                      </span>
                    </button>
                  ))}
                </div>

                <AnimatePresence mode="wait">
                  {entryMode === 'upload' ? (
                    <motion.div key="upload" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
                      {/* Drag-and-drop zone */}
                      <div
                        onDragOver={e => { e.preventDefault(); setDragging(true); }}
                        onDragLeave={() => setDragging(false)}
                        onDrop={handleDrop}
                        className={`border-2 border-dashed rounded-2xl p-8 flex flex-col items-center text-center transition-colors ${dragging ? 'border-white/40 bg-white/5' : 'border-white/10 hover:border-white/20'}`}
                      >
                        <UploadCloud className={`w-8 h-8 mb-3 transition-colors ${dragging ? 'text-white' : 'text-gray-600'}`} />
                        <p className="text-white text-sm font-medium mb-1">Drop your CV here</p>
                        <p className="text-gray-500 text-xs mb-4">or browse to select a PDF</p>
                        <input type="file" className="hidden" id="resume-upload" accept=".pdf,.doc,.docx" onChange={e => setFile(e.target.files?.[0] || null)} />
                        <label htmlFor="resume-upload" className="text-xs font-bold uppercase tracking-widest cursor-pointer bg-white/5 hover:bg-white/10 border border-white/10 text-white px-4 py-2 rounded-lg transition-colors">
                          Browse
                        </label>
                      </div>

                      {file && (
                        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="mt-4 flex items-center justify-between bg-white/5 border border-white/10 rounded-xl px-4 py-3">
                          <div className="flex items-center gap-2">
                            <FileText className="w-4 h-4 text-white/50" />
                            <span className="text-xs text-white truncate max-w-[150px]">{file.name}</span>
                          </div>
                          <button onClick={() => setFile(null)} className="text-gray-500 hover:text-white transition-colors">
                            <X className="w-3.5 h-3.5" />
                          </button>
                        </motion.div>
                      )}

                      {parsedData && (
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-4 bg-white/[0.03] border border-white/10 rounded-xl p-4 space-y-3">
                          <p className="text-[10px] uppercase tracking-widest text-gray-400 font-bold">Extracted Vectors</p>
                          <div className="flex flex-wrap gap-1.5">
                            {parsedData.skills?.map((s: string) => (
                              <span key={s} className="px-2 py-0.5 bg-white/5 border border-white/10 rounded-md text-xs text-gray-300">{s}</span>
                            ))}
                          </div>
                        </motion.div>
                      )}

                      <button
                        disabled={!file || parsing}
                        onClick={handleParseResume}
                        className="w-full mt-5 bg-white hover:bg-gray-200 text-black font-bold py-3 rounded-xl text-sm transition-all disabled:opacity-40 flex items-center justify-center gap-2"
                      >
                        {parsing ? <><Sparkles className="w-4 h-4 animate-pulse" /> Parsing...</> : 'Parse Resume'}
                      </button>

                      {parsedData && (
                        <form onSubmit={handleSaveProfile} className="mt-4 space-y-3">
                          <div>
                            <label className={labelCls}>Description (for GNN Text Channel)</label>
                            <textarea rows={3} value={editData.description} onChange={e => setEditData({ ...editData, description: e.target.value })} className={inputCls + ' resize-none'} />
                          </div>
                          <button type="submit" disabled={saving} className="w-full bg-white hover:bg-gray-200 text-black font-bold py-3 rounded-xl text-sm transition-all disabled:opacity-40 flex items-center justify-center gap-2">
                            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Save to Database'}
                          </button>
                        </form>
                      )}
                    </motion.div>
                  ) : (
                    /* === MANUAL FORM === */
                    <motion.form key="manual" onSubmit={handleSaveProfile} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} className="space-y-4">
                      <div>
                        <label className={labelCls}>Professional Summary</label>
                        <textarea required rows={3} value={editData.description} onChange={e => setEditData({ ...editData, description: e.target.value })} placeholder="Dense text for the GNN sentence transformer channel..." className={inputCls + ' resize-none'} />
                      </div>
                      <div>
                        <label className={labelCls}>Skills (Comma separated)</label>
                        <input required type="text" value={editData.skills} onChange={e => setEditData({ ...editData, skills: e.target.value })} placeholder="React, PyTorch, Python..." className={inputCls} />
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className={labelCls}>Experience</label>
                          <select value={editData.experience_level} onChange={e => setEditData({ ...editData, experience_level: e.target.value })} className={inputCls + ' appearance-none'}>
                            <option value="entry">Entry</option>
                            <option value="junior">Junior (0-2)</option>
                            <option value="mid">Mid (3-5)</option>
                            <option value="senior">Senior (5+)</option>
                            <option value="lead">Lead</option>
                            <option value="principal">Principal</option>
                          </select>
                        </div>
                        <div>
                          <label className={labelCls}>Contract</label>
                          <select value={editData.contract_preference} onChange={e => setEditData({ ...editData, contract_preference: e.target.value })} className={inputCls + ' appearance-none'}>
                            <option value="permanent">Permanent</option>
                            <option value="contract">Contract</option>
                          </select>
                        </div>
                      </div>
                      <div>
                        <label className={labelCls}>Current Salary (USD)</label>
                        <input type="number" value={editData.salary_current} onChange={e => setEditData({ ...editData, salary_current: e.target.value })} placeholder="e.g. 120000" className={inputCls} />
                      </div>
                      <button type="submit" disabled={saving} className="w-full bg-white hover:bg-gray-200 text-black font-bold py-3 rounded-xl text-sm transition-all disabled:opacity-40 flex items-center justify-center gap-2">
                        {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Save Profile'}
                      </button>
                    </motion.form>
                  )}
                </AnimatePresence>
              </motion.div>
            ) : editing ? (
              /* === EDIT MODE === */
              <motion.form key="edit" onSubmit={handleSaveProfile} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-4 border-t border-white/10 pt-5">
                <div>
                  <label className={labelCls}>Professional Summary</label>
                  <textarea rows={3} value={editData.description} onChange={e => setEditData({ ...editData, description: e.target.value })} className={inputCls + ' resize-none'} />
                </div>
                <div>
                  <label className={labelCls}>Skills (CSV)</label>
                  <input type="text" value={editData.skills} onChange={e => setEditData({ ...editData, skills: e.target.value })} className={inputCls} />
                </div>
                <div>
                  <label className={labelCls}>Target Salary</label>
                  <input type="number" value={editData.salary_current} onChange={e => setEditData({ ...editData, salary_current: e.target.value })} className={inputCls} />
                </div>
                <button type="submit" disabled={saving} className="w-full bg-white hover:bg-gray-200 text-black font-bold py-2.5 rounded-xl text-sm flex items-center justify-center gap-2">
                  {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <><Save className="w-3.5 h-3.5" /> Sync Profile</>}
                </button>
              </motion.form>
            ) : (
              /* === PROFILE VIEW === */
              <motion.div key="view" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-4">
                {profile?.description && (
                  <div className="border-t border-white/10 pt-4">
                    <label className={labelCls}>Summary</label>
                    <p className="text-gray-300 text-xs leading-relaxed line-clamp-3">{profile.description}</p>
                  </div>
                )}
                <div className="border-t border-white/10 pt-4">
                  <label className={labelCls}>Experience Bucket</label>
                  <div className="text-sm font-medium text-white capitalize">{profile?.experience_bucket}</div>
                </div>
                <div className="border-t border-white/10 pt-4">
                  <label className={labelCls}>Contract</label>
                  <div className="text-sm font-medium text-white capitalize">{profile?.contract_preference}</div>
                </div>
                <div className="border-t border-white/10 pt-4">
                  <label className={labelCls}>Skills</label>
                  <div className="flex flex-wrap gap-1.5 mt-1">
                    {profile?.skills?.map((s: string) => (
                      <span key={s} className="px-2 py-0.5 bg-white/5 border border-white/10 text-xs text-gray-300 rounded-md">{s}</span>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* === RIGHT: Recommendations Feed === */}
        <div className="lg:col-span-2">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-white flex items-center gap-3">
              <Briefcase className="w-5 h-5 text-gray-400" />
              Matched Positions
            </h2>
            {recommendations.length > 0 && (
              <span className="text-[10px] font-bold text-white uppercase tracking-widest bg-white/5 border border-white/10 px-3 py-1.5 rounded-full">
                {recommendations.length} Results
              </span>
            )}
          </div>

          {interactionMessage && (
            <div className="mb-4 rounded-xl border border-green-500/30 bg-green-500/10 px-4 py-2 text-xs text-green-300">
              {interactionMessage}
            </div>
          )}

          <div className="space-y-3">
            {!setupComplete ? (
              <div className="border border-white/5 bg-[#050505]/50 p-12 rounded-3xl text-center">
                <Briefcase className="w-8 h-8 text-gray-700 mx-auto mb-4" />
                <h3 className="text-white font-medium mb-1">Complete your profile</h3>
                <p className="text-sm text-gray-500">Upload your CV or fill the manual form to activate GNN matching.</p>
              </div>
            ) : loadingRecommendations ? (
              <div className="border border-white/5 bg-[#050505]/50 p-12 rounded-3xl text-center">
                <Loader2 className="w-7 h-7 animate-spin text-gray-700 mx-auto mb-4" />
                <h3 className="text-white font-medium">Scanning graph...</h3>
                <p className="text-sm text-gray-500">Retrieving personalized recommendations</p>
              </div>
            ) : recommendations.length === 0 ? (
              <div className="border border-white/5 bg-[#050505]/50 p-12 rounded-3xl text-center">
                <Briefcase className="w-8 h-8 text-gray-700 mx-auto mb-4" />
                <h3 className="text-white font-medium">No positions posted</h3>
                <p className="text-sm text-gray-500">Check back soon.</p>
              </div>
            ) : (
              recommendations.map((job, i) => (
                <motion.div
                  key={job.job_ID || job.id}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.07 }}
                  className="group bg-[#050505] border border-white/10 p-5 rounded-2xl hover:bg-white/[0.025] transition-all relative overflow-hidden cursor-pointer"
                  onClick={() => setSelectedJob(job)}
                >
                  <div className="absolute inset-y-0 left-0 w-[2px] bg-white opacity-0 group-hover:opacity-100 transition-opacity" />
                  <div className="flex justify-between items-start mb-3 gap-3">
                    <div>
                      <h3 className="text-base font-bold text-white tracking-tight">{job.title}</h3>
                      <p className="text-xs text-gray-500 font-mono mt-0.5">{job.company} · {job.employmentType}</p>
                    </div>
                    <div className="bg-white/5 border border-white/10 px-3 py-1 rounded-full">
                      <span className="text-white text-[10px] font-bold uppercase tracking-widest">{job.match_score != null ? `${Math.round(job.match_score * 100)}% Match` : 'Match'}</span>
                    </div>
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed line-clamp-2 mb-3 min-h-9">
                    {job.description || 'No additional description available.'}
                  </p>
                  <div className="flex justify-between items-center border-t border-white/5 pt-3 mt-2 gap-3">
                    <div className="flex gap-1.5">
                      {job.skills?.slice(0, 3).map((s: string) => (
                        <span key={s} className="text-[10px] text-gray-500 bg-white/5 px-2 py-1 rounded-md">{s}</span>
                      ))}
                    </div>
                    {appliedJobs.has(job.job_ID || job.id) ? (
                      <span className="text-green-400 text-[10px] font-bold uppercase tracking-widest flex items-center gap-1 bg-green-500/10 border border-green-500/20 px-3 py-1.5 rounded-lg">
                        <CheckCircle2 className="w-3 h-3" /> Applied
                      </span>
                    ) : (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleApply(job.job_ID || job.id);
                        }}
                        className="text-white text-[10px] font-bold uppercase tracking-widest flex items-center gap-1 bg-white/5 hover:bg-white/10 border border-white/10 px-3 py-1.5 rounded-lg transition-colors"
                      >
                        Apply <ChevronRight className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </motion.div>
              ))
            )}
          </div>
        </div>
      </div>

      <AnimatePresence>
        {selectedJob && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 z-[60] flex items-start justify-center p-4 pt-24"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: 16 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 16 }}
              className="bg-[#0a0a0a] border border-white/10 p-8 rounded-3xl w-full max-w-2xl relative max-h-[calc(100vh-7rem)] overflow-y-auto"
            >
              <button
                onClick={() => setSelectedJob(null)}
                className="absolute top-5 right-5 text-gray-500 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>

              <h3 className="text-2xl font-bold text-white mb-1">{selectedJob.title || 'Recommended Job'}</h3>
              <p className="text-sm text-gray-400 mb-5">
                {selectedJob.company || 'Unknown Company'} · {selectedJob.employmentType || 'N/A'}
              </p>

              <div className="grid sm:grid-cols-2 gap-4 mb-5">
                <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                  <div className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">Match</div>
                  <div className="text-white font-bold text-lg">
                    {selectedJob.match_score != null ? `${Math.round(selectedJob.match_score * 100)}%` : 'N/A'}
                  </div>
                </div>
                <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                  <div className="text-[10px] uppercase tracking-widest text-gray-500 mb-1">Experience</div>
                  <div className="text-white font-bold text-lg capitalize">{selectedJob.experience || 'N/A'}</div>
                </div>
              </div>

              <div className="mb-5">
                <div className="text-[10px] uppercase tracking-widest text-gray-500 mb-2">Description</div>
                <p className="text-sm text-gray-300 leading-relaxed max-h-40 overflow-y-auto pr-2">
                  {selectedJob.description || 'No job description available.'}
                </p>
              </div>

              <div className="mb-6">
                <div className="text-[10px] uppercase tracking-widest text-gray-500 mb-2">Skills</div>
                <div className="flex flex-wrap gap-2">
                  {(selectedJob.skills || []).map((skill: string) => (
                    <span key={skill} className="px-2.5 py-1 rounded-md border border-white/10 bg-white/5 text-xs text-gray-300">
                      {skill}
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex justify-end gap-3">
                <button
                  onClick={() => setSelectedJob(null)}
                  className="px-4 py-2 text-sm border border-white/10 rounded-lg text-gray-300 hover:text-white hover:bg-white/5"
                >
                  Close
                </button>
                {appliedJobs.has(selectedJob.job_ID || selectedJob.id) ? (
                  <span className="text-green-400 text-xs font-bold uppercase tracking-widest flex items-center gap-1 bg-green-500/10 border border-green-500/20 px-4 py-2 rounded-lg">
                    <CheckCircle2 className="w-3 h-3" /> Applied
                  </span>
                ) : (
                  <button
                    onClick={() => handleApply(selectedJob.job_ID || selectedJob.id)}
                    className="px-4 py-2 text-sm border border-white/10 rounded-lg text-white bg-white/10 hover:bg-white/15"
                  >
                    Apply Now
                  </button>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
