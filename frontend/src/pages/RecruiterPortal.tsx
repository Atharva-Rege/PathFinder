import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Settings, Loader2, Save, X, Network, Briefcase, Mail, Phone } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function RecruiterPortal() {
  const [profile, setProfile] = useState<any>(null);
  const [candidates, setCandidates] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [selectedCandidate, setSelectedCandidate] = useState<any>(null);
  const navigate = useNavigate();

  const handleContact = async (c: any) => {
    setSelectedCandidate(c);
    const token = localStorage.getItem('token');
    if (!token) return;
    try {
      await fetch('http://127.0.0.1:8000/api/log-interaction/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Token ${token}` },
        body: JSON.stringify({ candidate_id: c.candidate_ID || c.id, action: 'shortlist' })
      });
    } catch(err) { console.error('Failed to log interaction', err); }
  };

  const [formData, setFormData] = useState({
      skills: '', salary: '', title: '', description: ''
  });

  const fetchProfile = async () => {
      const token = localStorage.getItem('token');
      if(!token) return navigate('/auth');
      try {
          const res = await fetch('http://127.0.0.1:8000/api/auth/me/', {
              headers: { 'Authorization': `Token ${token}` }
          });
          if(res.ok) {
              const data = await res.json();
              if(data.role !== 'recruiter') return navigate('/auth');
              setProfile(data.profile);
              setFormData({
                  skills: data.profile.skills.join(', '),
                  salary: data.profile.salary,
                  title: data.profile.title,
                  description: data.profile.description
              });
              
              setLoadingRecommendations(true);
              const candRes = await fetch('http://127.0.0.1:8000/api/recommend/candidates/', {
                  headers: { 'Authorization': `Token ${token}` }
              });
              const candData = await candRes.json();
              setCandidates(candData.recommendations || []);
              setLoadingRecommendations(false);
          } else {
              localStorage.removeItem('token');
              navigate('/auth');
          }
      } catch(err) { console.error(err); }
      setLoading(false);
  };

  useEffect(() => {
    fetchProfile();
  }, [navigate]);

  const handleUpdate = async (e: React.FormEvent) => {
      e.preventDefault();
      setSaving(true);
      const token = localStorage.getItem('token');
      try {
          await fetch('http://127.0.0.1:8000/api/auth/update/', {
              method: 'PUT',
              headers: { 
                  'Content-Type': 'application/json',
                  'Authorization': `Token ${token}`
              },
              body: JSON.stringify({
                  title: formData.title,
                  description: formData.description,
                  skills: formData.skills.split(',').map(s=>s.trim()),
                  salary: formData.salary
              })
          });
          await fetchProfile();
          setEditing(false);
      } catch(e) { console.error(e); }
      setSaving(false);
  };

  if (loading) return <div className="min-h-screen flex items-center justify-center"><Loader2 className="w-8 h-8 animate-spin text-white" /></div>;

  return (
        <div className="max-w-7xl mx-auto px-4 pt-6 pb-4 h-[calc(100vh-80px)] overflow-hidden flex flex-col">
            <div className="flex items-center justify-between mb-6 shrink-0">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white">Recruiter Dashboard</h1>
          <p className="text-gray-500 text-sm mt-1">Manage your job posting and review matched candidates</p>
        </div>
      </div>

            <div className="grid lg:grid-cols-3 gap-8 items-start flex-1 min-h-0">
        {/* Profile Sidebar */}
                <div className="lg:col-span-1 border border-white/10 bg-[#050505] rounded-3xl p-6 relative overflow-y-auto group h-full">
            <div className="absolute top-0 right-0 p-3">
                <button onClick={()=>setEditing(!editing)} className="p-2 bg-white/5 hover:bg-white/10 rounded-xl transition-colors">
                    {editing ? <X className="w-4 h-4 text-white" /> : <Settings className="w-4 h-4 text-gray-400" />}
                </button>
            </div>
            
            <div className="flex items-center mb-6 pt-2">
                <div className="w-12 h-12 bg-white/10 rounded-full flex items-center justify-center border border-white/10">
                    <Briefcase className="w-5 h-5 text-white" />
                </div>
                <div className="ml-4">
                    <h2 className="text-lg font-bold text-white tracking-tight">{profile?.company}</h2>
                    <p className="text-xs text-gray-500">Hiring Manager</p>
                </div>
            </div>

            <AnimatePresence mode="wait">
                {editing ? (
                    <motion.form 
                        key="edit" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}
                        onSubmit={handleUpdate} className="space-y-4 border-t border-white/10 pt-6"
                    >
                        <div>
                            <label className="text-[10px] uppercase text-gray-500 font-bold block mb-1">Job Title</label>
                            <input type="text" value={formData.title} onChange={e=>setFormData({...formData, title: e.target.value})} className="w-full bg-white/5 border border-white/10 rounded-lg py-2 px-3 text-white text-sm focus:outline-none focus:ring-1 focus:ring-white" />
                        </div>
                        <div>
                            <label className="text-[10px] uppercase text-gray-500 font-bold block mb-1">Required Skills (comma separated)</label>
                            <input type="text" value={formData.skills} onChange={e=>setFormData({...formData, skills: e.target.value})} className="w-full bg-white/5 border border-white/10 rounded-lg py-2 px-3 text-white text-sm focus:outline-none focus:ring-1 focus:ring-white" />
                        </div>
                        <div>
                            <label className="text-[10px] uppercase text-gray-500 font-bold block mb-1">Salary Budget</label>
                            <input type="number" value={formData.salary} onChange={e=>setFormData({...formData, salary: e.target.value})} className="w-full bg-white/5 border border-white/10 rounded-lg py-2 px-3 text-white text-sm focus:outline-none focus:ring-1 focus:ring-white" />
                        </div>
                        <button disabled={saving} type="submit" className="w-full bg-white hover:bg-gray-200 text-black py-2.5 rounded-lg text-sm font-bold flex justify-center items-center mt-4">
                            {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <><Save className="w-4 h-4 mr-2" /> Save Changes</>}
                        </button>
                    </motion.form>
                ) : (
                    <motion.div key="view" initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} className="space-y-4">
                        <div className="border-t border-white/10 pt-4">
                            <label className="text-[10px] uppercase tracking-widest text-gray-500 font-bold block mb-2">Open Position</label>
                            <div className="text-sm font-medium text-white">{profile?.title}</div>
                        </div>
                        <div className="border-t border-white/10 pt-4">
                            <label className="text-[10px] uppercase tracking-widest text-gray-500 font-bold block mb-3">Required Skills</label>
                            <div className="flex flex-wrap gap-2">
                                {profile?.skills?.map((s: string) => (
                                    <span key={s} className="px-2.5 py-1 bg-white/5 text-gray-300 text-xs rounded-md border border-white/10">{s}</span>
                                ))}
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>

        {/* Feed */}
        <div className="lg:col-span-2 flex flex-col h-full lg:pl-4 min-h-0">
            <div className="flex items-center justify-between mb-6 shrink-0">
                <h2 className="text-xl font-bold flex items-center text-white tracking-tight">
                    <Network className="w-5 h-5 mr-3 text-gray-400" />
                    Recommended Candidates
                </h2>
                {candidates.length > 0 && <span className="text-[10px] font-bold text-white uppercase tracking-widest bg-white/10 px-3 py-1.5 rounded-full border border-white/10">{candidates.length} Matches</span>}
            </div>
            
            <div className="space-y-3 overflow-y-auto pr-1 flex-1 min-h-0">
                {loadingRecommendations ? (
                    <div className="border border-white/5 bg-[#050505] p-10 rounded-3xl text-center">
                        <Loader2 className="w-8 h-8 animate-spin text-gray-700 mx-auto mb-4" />
                        <h3 className="text-white font-medium">Scanning graph...</h3>
                        <p className="text-sm text-gray-500 mt-1">Retrieving candidate matches</p>
                    </div>
                ) : candidates.length === 0 ? (
                    <div className="border border-white/5 bg-[#050505] p-10 rounded-3xl text-center">
                        <Network className="w-8 h-8 text-gray-700 mx-auto mb-4" />
                        <h3 className="text-white font-medium">No candidates yet</h3>
                        <p className="text-sm text-gray-500 mt-1">Matches will appear here as candidates sign up.</p>
                    </div>
                ) : candidates.map((c, i) => (
                    <motion.div 
                        key={c.candidate_ID || c.id}
                        initial={{opacity:0, y:20}} animate={{opacity:1, y:0}} 
                        transition={{delay: i * 0.1}}
                        className="group bg-[#050505] border border-white/10 p-5 rounded-2xl hover:bg-white/[0.03] transition-colors overflow-hidden relative cursor-pointer"
                        onClick={() => setSelectedCandidate(c)}
                    >
                        <div className="absolute inset-y-0 left-0 w-1 bg-white opacity-0 group-hover:opacity-100 transition-opacity"></div>
                        <div className="flex justify-between items-start mb-2 gap-3">
                            <div>
                                <h3 className="text-lg font-bold text-white tracking-tight">{c.name}</h3>
                                <p className="text-gray-400 text-xs mt-1 capitalize">{c.experience_bucket} level</p>
                            </div>
                            {c.match_score != null && (
                                <div className="bg-white/5 border border-white/10 px-3 py-1 rounded-full">
                                    <span className="text-white text-[10px] font-bold uppercase tracking-widest">{Math.round(c.match_score * 100)}% Match</span>
                                </div>
                            )}
                        </div>
                        <p className="text-xs text-gray-400 leading-relaxed line-clamp-2 mb-3 min-h-9">
                          {c.description || 'No profile summary provided.'}
                        </p>
                        <div className="flex justify-between items-center border-t border-white/5 pt-3 mt-2 gap-3">
                            <div className="flex gap-2">
                                {c.skills?.slice(0, 3).map((s: string) => (
                                    <span key={s} className="text-[10px] text-gray-500 bg-white/5 px-2 py-1 rounded-md border border-white/5">{s}</span>
                                ))}
                            </div>
                            <button 
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleContact(c);
                                }}
                                className="text-white text-xs font-bold uppercase tracking-wider bg-white/10 px-3 py-1.5 rounded-lg border border-white/10 hover:bg-white/15 transition-colors"
                            >
                                Contact
                            </button>
                        </div>
                    </motion.div>
                ))}
            </div>
        </div>
      </div>

      {/* Contact Modal */}
      <AnimatePresence>
        {selectedCandidate && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/80 z-[60] flex items-start justify-center p-4 pt-24"
          >
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }} 
              animate={{ opacity: 1, scale: 1, y: 0 }} 
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            className="bg-[#0a0a0a] border border-white/10 p-8 rounded-3xl w-full max-w-2xl relative max-h-[calc(100vh-7rem)] overflow-y-auto"
            >
              <button onClick={() => setSelectedCandidate(null)} className="absolute top-5 right-5 text-gray-500 hover:text-white transition-colors">
                <X className="w-5 h-5" />
              </button>
              
              <h3 className="text-2xl font-bold text-white mb-2">Contact {selectedCandidate.name}</h3>
              <p className="text-sm text-gray-500 mb-6">{selectedCandidate.experience_bucket} level candidate</p>

                            <div className="grid sm:grid-cols-2 gap-3 mb-5">
                                <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                                    <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1">Match</div>
                                    <div className="text-white font-semibold">
                                        {selectedCandidate.match_score != null ? `${Math.round(selectedCandidate.match_score * 100)}%` : 'N/A'}
                                    </div>
                                </div>
                                <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                                    <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1">Source</div>
                                    <div className="text-white font-semibold capitalize">{selectedCandidate.source || 'N/A'}</div>
                                </div>
                            </div>

                            <div className="mb-5">
                                <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2">Profile Summary</div>
                                <p className="text-sm text-gray-300 leading-relaxed max-h-28 overflow-y-auto pr-2">
                                    {selectedCandidate.description || 'No summary provided.'}
                                </p>
                            </div>

                            <div className="mb-6">
                                <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2">Skills</div>
                                <div className="flex flex-wrap gap-2">
                                    {(selectedCandidate.skills || []).map((skill: string) => (
                                        <span key={skill} className="px-2.5 py-1 bg-white/5 border border-white/10 rounded-md text-xs text-gray-300">
                                            {skill}
                                        </span>
                                    ))}
                                </div>
                            </div>
              
                            <div className="grid sm:grid-cols-2 gap-4">
                <a href={`mailto:${selectedCandidate.email}`} className="flex items-center gap-4 p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors group">
                  <div className="p-3 bg-white/10 rounded-lg group-hover:bg-white/20 transition-colors">
                    <Mail className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1">Email via Platform</div>
                    <div className="text-white font-medium">{selectedCandidate.email || "No email provided"}</div>
                  </div>
                </a>
                
                <a href={`tel:${selectedCandidate.phone}`} className="flex items-center gap-4 p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors group">
                  <div className="p-3 bg-white/10 rounded-lg group-hover:bg-white/20 transition-colors">
                    <Phone className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-1">Direct Phone</div>
                    <div className="text-white font-medium">{selectedCandidate.phone || "No phone provided"}</div>
                  </div>
                </a>
              </div>
              
              <button onClick={() => setSelectedCandidate(null)} className="w-full mt-8 bg-white text-black font-bold py-3.5 rounded-xl hover:bg-gray-200 transition-colors">
                Done
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
