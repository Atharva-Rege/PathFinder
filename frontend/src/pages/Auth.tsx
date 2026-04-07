import { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Mail, Lock, User, ArrowRight, BrainCircuit, Loader2, ArrowLeft,
  UploadCloud, FileText, X, Keyboard, AlertCircle, CheckCircle2, Plus
} from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';

const API = 'http://127.0.0.1:8000/api';
const CONTRACT_OPTIONS = ['permanent', 'contract', 'freelance', 'temporary'] as const;
const EXPERIENCE_OPTIONS = ['entry', 'junior', 'mid', 'senior', 'lead', 'principal'] as const;
const ORIGIN_OPTIONS = ['direct', 'linkedin', 'indeed', 'glassdoor', 'referral'] as const;
const EMPLOY_OPTIONS = ['permanent', 'contract', 'freelance', 'temporary'] as const;

const SKILL_SUGGESTIONS = [
  'Python', 'JavaScript', 'TypeScript', 'React', 'Node.js', 'Django', 'FastAPI',
  'AWS', 'GCP', 'Azure', 'Docker', 'Kubernetes', 'PostgreSQL', 'MongoDB', 'Redis',
  'PyTorch', 'TensorFlow', 'scikit-learn', 'Pandas', 'NumPy', 'SQL', 'GraphQL',
  'Next.js', 'Flutter', 'Swift', 'Kotlin', 'Java', 'Go', 'Rust', 'C++', 'C#', 'Git',
  'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision', 'Data Science',
  'System Design', 'Leadership', 'Product Management', 'UI/UX', 'Figma',
  'HTML', 'CSS', 'Vue', 'Angular', 'Express', 'Spring', 'Flask', 'Spark', 'Kafka', 'Terraform',
];

const inputCls = 'w-full bg-white/5 border border-white/10 rounded-xl py-3 pl-12 pr-4 text-white placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-white transition-all';
const fieldCls = 'w-full bg-white/5 border border-white/10 rounded-lg py-2 px-3 text-white text-sm focus:outline-none focus:ring-1 focus:ring-white';
const labelCls = 'text-[10px] uppercase tracking-widest text-gray-500 font-bold block mb-1';

interface CandidateData {
  public_name: string; profile_description_text: string; skills: string[];
  experience_level: string; contract_preference: string; origin_source: string;
  salary_current: string; phone: string;
}
interface JobData {
  title: string; company_owner: string; job_description_text: string; skills: string[];
  employment_type: string; experience_required: string; category: string; salary: string;
}

// ─── Shows yellow tag only if field is genuinely empty after extraction ───
const needsReview = (ext: Record<string, any> | null, key: string, val: any) => {
  if (!ext || !ext[key]) return false;
  if (Array.isArray(val) && val.length > 0) return false;
  if (typeof val === 'string' && val.trim()) return false;
  if (typeof val === 'number') return false;
  const v = ext[key].value;
  return v === null || v === '' || (Array.isArray(v) && v.length === 0);
};

// ═════════════════════════════════════
// SkillSelector — top-level so it never remounts
// ═════════════════════════════════════
function SkillSelector({ selected, onChange, warn }: { selected: string[]; onChange: (s: string[]) => void; warn?: boolean }) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLInputElement>(null);

  const filtered = SKILL_SUGGESTIONS.filter(s => s.toLowerCase().includes(query.toLowerCase()) && !selected.includes(s)).slice(0, 8);
  const add = (s: string) => { const t = s.trim(); if (t && !selected.includes(t)) onChange([...selected, t]); setQuery(''); setOpen(false); ref.current?.focus(); };
  const remove = (s: string) => onChange(selected.filter(x => x !== s));
  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if ((e.key === 'Enter' || e.key === ',') && query.trim()) { e.preventDefault(); add(query.replace(',', '').trim()); }
    if (e.key === 'Backspace' && !query && selected.length) remove(selected[selected.length - 1]);
  };

  return (
    <div className="relative">
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-2">
          {selected.map(s => (
            <span key={s} className="flex items-center gap-1 px-2.5 py-1 bg-white/10 border border-white/15 rounded-lg text-xs text-white font-medium">
              {s}
              <button type="button" onClick={() => remove(s)} className="text-gray-400 hover:text-white ml-0.5"><X className="w-3 h-3" /></button>
            </span>
          ))}
        </div>
      )}
      <div className={`flex items-center gap-2 ${fieldCls} ${warn && selected.length === 0 ? 'border-yellow-500/40 bg-yellow-500/5' : ''}`}
        onClick={() => { setOpen(true); ref.current?.focus(); }}>
        <input ref={ref} type="text" value={query}
          onChange={e => { setQuery(e.target.value); setOpen(true); }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={onKey}
          placeholder={selected.length === 0 ? 'Search or type a skill, press Enter...' : 'Add more...'}
          className="flex-1 bg-transparent outline-none text-sm text-white placeholder-gray-600 min-w-0" />
        {query.trim() && <button type="button" onClick={() => add(query)} className="text-gray-400 hover:text-white shrink-0"><Plus className="w-3.5 h-3.5" /></button>}
      </div>
      <AnimatePresence>
        {open && (filtered.length > 0 || query.trim()) && (
          <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={{ duration: 0.12 }}
            className="absolute z-50 top-full left-0 right-0 mt-1 bg-[#0a0a0a] border border-white/10 rounded-xl overflow-hidden shadow-2xl">
            {filtered.map(s => (
              <button key={s} type="button" onMouseDown={() => add(s)} className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-white/5 hover:text-white transition-colors">{s}</button>
            ))}
            {query.trim() && !SKILL_SUGGESTIONS.map(s => s.toLowerCase()).includes(query.trim().toLowerCase()) && (
              <button type="button" onMouseDown={() => add(query)} className="w-full text-left px-3 py-2 text-sm text-white border-t border-white/5 hover:bg-white/5 flex items-center gap-2">
                <Plus className="w-3.5 h-3.5 text-gray-400" />Add "{query.trim()}"
              </button>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ═════════════════════════════════════
// CandidateForm — top-level so it never remounts
// ═════════════════════════════════════
function CandidateForm({ data, onChange, extraction, loading, onSubmit }:
  { data: CandidateData; onChange: (d: CandidateData) => void; extraction: Record<string, any> | null; loading: boolean; onSubmit: (e: React.FormEvent) => void }) {
  const R = ({ k, v }: { k: string; v: any }) => needsReview(extraction, k, v)
    ? <span className="ml-1.5 text-[9px] font-bold uppercase tracking-widest text-yellow-500">⚠ fill in</span> : null;

  return (
    <form onSubmit={onSubmit} className="space-y-4">
      {extraction && (
        <div className="flex items-start gap-2 bg-white/5 border border-white/10 rounded-xl p-3">
          <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
          <p className="text-xs text-gray-400">Details extracted. Fields marked <span className="text-yellow-400 font-semibold">⚠ fill in</span> could not be read — please complete them.</p>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className={labelCls}>Full Name <R k="public_name" v={data.public_name} /></label>
          <input required type="text" value={data.public_name} placeholder="Your full name"
            onChange={e => onChange({ ...data, public_name: e.target.value })}
            className={fieldCls + (!data.public_name && extraction ? ' border-yellow-500/40 bg-yellow-500/5' : '')} />
        </div>
        <div>
          <label className={labelCls}>Phone Number</label>
          <input type="tel" value={data.phone} placeholder="e.g. +91 98765 43210"
            onChange={e => onChange({ ...data, phone: e.target.value })} className={fieldCls} />
        </div>
        <div>
          <label className={labelCls}>Current Salary (leave blank if unsure)</label>
          <input type="number" value={data.salary_current} placeholder="e.g. 1200000"
            onChange={e => onChange({ ...data, salary_current: e.target.value })} className={fieldCls} />
        </div>
        <div>
          <label className={labelCls}>Experience Level</label>
          <select value={data.experience_level} onChange={e => onChange({ ...data, experience_level: e.target.value })} className={fieldCls + ' appearance-none'}>
            {EXPERIENCE_OPTIONS.map(v => <option key={v} value={v}>{v.charAt(0).toUpperCase() + v.slice(1)}</option>)}
          </select>
        </div>
        <div className="col-span-1 md:col-span-2">
          <label className={labelCls}>Professional Summary <R k="profile_description_text" v={data.profile_description_text} /></label>
          <textarea required rows={3} value={data.profile_description_text}
            placeholder="Describe your background, experience and goals..."
            onChange={e => onChange({ ...data, profile_description_text: e.target.value })}
            className={fieldCls + ' resize-none' + (!data.profile_description_text && extraction ? ' border-yellow-500/40 bg-yellow-500/5' : '')} />
        </div>
        <div className="col-span-1 md:col-span-2">
          <label className={labelCls}>Skills <R k="skills" v={data.skills} /></label>
          <SkillSelector selected={data.skills} onChange={s => onChange({ ...data, skills: s })} warn={data.skills.length === 0 && !!extraction} />
        </div>
        <div>
          <label className={labelCls}>Work Preference</label>
          <select value={data.contract_preference} onChange={e => onChange({ ...data, contract_preference: e.target.value })} className={fieldCls + ' appearance-none'}>
            {CONTRACT_OPTIONS.map(v => <option key={v} value={v}>{v.charAt(0).toUpperCase() + v.slice(1)}</option>)}
          </select>
        </div>
        <div>
          <label className={labelCls}>How did you find us?</label>
          <select value={data.origin_source} onChange={e => onChange({ ...data, origin_source: e.target.value })} className={fieldCls + ' appearance-none'}>
            {ORIGIN_OPTIONS.map(v => <option key={v} value={v}>{v.charAt(0).toUpperCase() + v.slice(1)}</option>)}
          </select>
        </div>
      </div>
      <button disabled={loading} type="submit"
        className="w-full bg-white hover:bg-gray-100 text-black font-bold py-3.5 rounded-xl transition-all flex items-center justify-center gap-2 mt-2">
        {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Create Account'}
      </button>
    </form>
  );
}

// ═════════════════════════════════════
// RecruiterForm — top-level
// ═════════════════════════════════════
function RecruiterForm({ data, onChange, extraction, loading, onSubmit }:
  { data: JobData; onChange: (d: JobData) => void; extraction: Record<string, any> | null; loading: boolean; onSubmit: (e: React.FormEvent) => void }) {
  const R = ({ k, v }: { k: string; v: any }) => needsReview(extraction, k, v)
    ? <span className="ml-1.5 text-[9px] font-bold uppercase tracking-widest text-yellow-500">⚠ fill in</span> : null;

  return (
    <form onSubmit={onSubmit} className="space-y-4">
      {extraction && (
        <div className="flex items-start gap-2 bg-white/5 border border-white/10 rounded-xl p-3">
          <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
          <p className="text-xs text-gray-400">Details extracted. Fields marked <span className="text-yellow-400 font-semibold">⚠ fill in</span> could not be read — please complete them.</p>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className={labelCls}>Job Title <R k="title" v={data.title} /></label>
          <input required type="text" value={data.title} placeholder="e.g. Senior Frontend Engineer"
            onChange={e => onChange({ ...data, title: e.target.value })}
            className={fieldCls + (!data.title && extraction ? ' border-yellow-500/40 bg-yellow-500/5' : '')} />
        </div>
        <div>
          <label className={labelCls}>Company Name</label>
          <input required type="text" value={data.company_owner} placeholder="e.g. Acme Corp"
            onChange={e => onChange({ ...data, company_owner: e.target.value })} className={fieldCls} />
        </div>
        <div className="col-span-1 md:col-span-2">
          <label className={labelCls}>Job Description <R k="job_description_text" v={data.job_description_text} /></label>
          <textarea required rows={3} value={data.job_description_text} placeholder="What does this role involve?"
            onChange={e => onChange({ ...data, job_description_text: e.target.value })}
            className={fieldCls + ' resize-none' + (!data.job_description_text && extraction ? ' border-yellow-500/40 bg-yellow-500/5' : '')} />
        </div>
        <div className="col-span-1 md:col-span-2">
          <label className={labelCls}>Required Skills <R k="skills" v={data.skills} /></label>
          <SkillSelector selected={data.skills} onChange={s => onChange({ ...data, skills: s })} warn={data.skills.length === 0 && !!extraction} />
        </div>
        <div>
          <label className={labelCls}>Employment Type</label>
          <select value={data.employment_type} onChange={e => onChange({ ...data, employment_type: e.target.value })} className={fieldCls + ' appearance-none'}>
            {EMPLOY_OPTIONS.map(v => <option key={v} value={v}>{v.charAt(0).toUpperCase() + v.slice(1)}</option>)}
          </select>
        </div>
        <div>
          <label className={labelCls}>Experience Required</label>
          <select value={data.experience_required} onChange={e => onChange({ ...data, experience_required: e.target.value })} className={fieldCls + ' appearance-none'}>
            {EXPERIENCE_OPTIONS.map(v => <option key={v} value={v}>{v.charAt(0).toUpperCase() + v.slice(1)}</option>)}
          </select>
        </div>
        <div>
          <label className={labelCls}>Category</label>
          <input type="text" value={data.category} placeholder="e.g. Engineering"
            onChange={e => onChange({ ...data, category: e.target.value })} className={fieldCls} />
        </div>
        <div>
          <label className={labelCls}>Salary Budget (leave blank if flexible)</label>
          <input type="number" value={data.salary} placeholder="e.g. 2000000"
            onChange={e => onChange({ ...data, salary: e.target.value })} className={fieldCls} />
        </div>
      </div>
      <button disabled={loading} type="submit"
        className="w-full bg-white hover:bg-gray-100 text-black font-bold py-3.5 rounded-xl transition-all flex items-center justify-center gap-2 mt-2">
        {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Create Account'}
      </button>
    </form>
  );
}

// ═════════════════════════════════════
// Main Auth component
// ═════════════════════════════════════
export default function Auth() {
  const [isLogin, setIsLogin] = useState(true);
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [role, setRole] = useState<'candidate' | 'recruiter'>('candidate');
  const [entryMode, setEntryMode] = useState<'upload' | 'manual'>('upload');
  const [loading, setLoading] = useState(false);
  const [errorMSG, setErrorMSG] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [parsing, setParsing] = useState(false);
  const [extraction, setExtraction] = useState<Record<string, any> | null>(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const mode = new URLSearchParams(location.search).get('mode');
    if (mode === 'signup') {
      setIsLogin(false);
      setStep(1);
      setErrorMSG('');
    } else if (mode === 'login') {
      setIsLogin(true);
      setStep(1);
      setErrorMSG('');
    }
  }, [location.search]);

  const [creds, setCreds] = useState({ username: '', email: '', password: '' });
  const [cand, setCand] = useState<CandidateData>({
    public_name: '', profile_description_text: '', skills: [],
    experience_level: 'mid', contract_preference: 'permanent',
    origin_source: 'direct', salary_current: '', phone: '',
  });
  const [job, setJob] = useState<JobData>({
    title: '', company_owner: '', job_description_text: '', skills: [],
    employment_type: 'permanent', experience_required: 'mid', category: '', salary: '',
  });

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault(); setLoading(true); setErrorMSG('');
    try {
      const res = await fetch(`${API}/auth/login/`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: creds.username, password: creds.password }) });
      const data = await res.json();
      if (res.ok) { localStorage.setItem('token', data.token); navigate(`/${data.role}`); }
      else setErrorMSG(data.error || 'Login failed.');
    } catch { setErrorMSG('Cannot reach server. Is the backend running?'); }
    setLoading(false);
  };

  const goToStep2 = () => {
    if (!creds.username.trim()) return setErrorMSG('Username is required.');
    if (!creds.email.trim()) return setErrorMSG('Email is required.');
    if (creds.password.length < 8) return setErrorMSG('Password must be at least 8 characters.');
    setErrorMSG(''); setStep(2);
  };

  const handleParse = async () => {
    if (!file) return;
    setParsing(true); setErrorMSG('');
    try {
      const fd = new FormData(); fd.append('file', file);
      const endpoint = role === 'candidate' ? 'parse-resume' : 'parse-job-pdf';
      const res = await fetch(`${API}/${endpoint}/`, { method: 'POST', body: fd });
      const data = await res.json();
      if (res.ok) {
        setExtraction(data.fields);
        if (role === 'candidate') {
          setCand(prev => ({
            ...prev,
            public_name: data.fields.public_name?.value || prev.public_name,
            profile_description_text: data.fields.profile_description_text?.value || prev.profile_description_text,
            skills: Array.isArray(data.fields.skills?.value) && data.fields.skills.value.length > 0 ? data.fields.skills.value : prev.skills,
            experience_level: data.fields.experience_level?.value || prev.experience_level,
            contract_preference: data.fields.contract_preference?.value || prev.contract_preference,
            origin_source: data.fields.origin_source?.value || prev.origin_source,
            salary_current: data.fields.salary_current?.value != null ? String(data.fields.salary_current.value) : prev.salary_current,
          }));
        } else {
          setJob(prev => ({
            ...prev,
            title: data.fields.title?.value || prev.title,
            job_description_text: data.fields.job_description_text?.value || prev.job_description_text,
            skills: Array.isArray(data.fields.skills?.value) && data.fields.skills.value.length > 0 ? data.fields.skills.value : prev.skills,
            employment_type: data.fields.employment_type?.value || prev.employment_type,
            experience_required: data.fields.experience_required?.value || prev.experience_required,
            category: data.fields.category?.value || prev.category,
            salary: data.fields.salary?.value != null ? String(data.fields.salary.value) : prev.salary,
          }));
        }
        setStep(3);
      } else setErrorMSG(data.error || 'Parsing failed. Try filling manually.');
    } catch { setErrorMSG('Cannot reach server. Is the backend running?'); }
    setParsing(false);
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault(); setLoading(true); setErrorMSG('');
    try {
      const base = { username: creds.username, email: creds.email, password: creds.password, role };
      const payload = role === 'candidate' ? {
        ...base,
        name: cand.public_name, phone: cand.phone,
        description: cand.profile_description_text, skills: cand.skills,
        experience_level: cand.experience_level, contract_preference: cand.contract_preference,
        origin_source: cand.origin_source,
        salary_current: cand.salary_current ? Number(cand.salary_current) : null,
      } : {
        ...base,
        title: job.title, company_owner: job.company_owner,
        job_description_text: job.job_description_text, skills: job.skills,
        employment_type: job.employment_type, experience_required: job.experience_required,
        category: job.category, salary: job.salary ? Number(job.salary) : null,
      };
      const res = await fetch(`${API}/auth/register/`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const data = await res.json();
      if (res.ok) { localStorage.setItem('token', data.token); navigate(`/${role}`); }
      else setErrorMSG(data.error || 'Registration failed.');
    } catch { setErrorMSG('Cannot reach server. Is the backend running?'); }
    setLoading(false);
  };

  const reset = () => {
    const nextMode = isLogin ? 'signup' : 'login';
    setIsLogin(!isLogin);
    setStep(1);
    setErrorMSG('');
    setExtraction(null);
    setFile(null);
    navigate(`/auth?mode=${nextMode}`, { replace: true });
  };

  return (
    <div className="min-h-[calc(100vh-80px)] flex flex-col items-center justify-center px-4 py-12">
      <motion.div layout initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="w-full max-w-2xl">
        <div className="bg-[#050505] border border-white/10 p-8 rounded-[24px] shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between mb-7">
            {(!isLogin && step > 1) ? (
              <button onClick={() => { setStep(step === 3 ? 2 : 1); setExtraction(null); setFile(null); }}
                className="text-gray-400 hover:text-white flex items-center text-sm font-medium transition-colors">
                <ArrowLeft className="w-4 h-4 mr-1" /> Back
              </button>
            ) : <div className="w-14" />}
            <div className="p-2.5 bg-white/5 rounded-xl border border-white/10"><BrainCircuit className="w-6 h-6 text-white" /></div>
            <div className="w-14" />
          </div>

          {!isLogin && (
            <div className="flex items-center justify-center gap-2 mb-6">
              {[1, 2, 3].map(s => <div key={s} className={`h-1 rounded-full transition-all duration-300 ${step >= s ? 'bg-white w-8' : 'bg-white/10 w-4'}`} />)}
            </div>
          )}

          <h2 className="text-2xl font-bold text-center text-white mb-1 tracking-tight">
            {isLogin ? 'Welcome back' : step === 1 ? 'Create your account' : step === 2 ? `Upload your ${role === 'candidate' ? 'CV / Resume' : 'Job Description'}` : 'Review your details'}
          </h2>
          <p className="text-center text-gray-500 mb-7 text-sm">
            {isLogin ? 'Sign in to access your Pathfinder dashboard'
              : step === 1 ? 'Start with your basic account information'
                : step === 2 ? 'Upload a PDF to auto-fill, or fill in manually'
                  : 'Verify your information before creating your account'}
          </p>

          {errorMSG && (
            <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 text-red-400 text-sm py-2.5 px-4 rounded-lg mb-5">
              <AlertCircle className="w-4 h-4 shrink-0" />{errorMSG}
            </div>
          )}

          <AnimatePresence mode="wait">
            {/* LOGIN */}
            {isLogin && (
              <motion.form key="login" onSubmit={handleLogin} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-4">
                <div className="relative group/i">
                  <User className="absolute left-4 top-3.5 w-5 h-5 text-gray-600 group-focus-within/i:text-white transition-colors" />
                  <input type="text" required autoComplete="username" placeholder="Username"
                    value={creds.username} onChange={e => setCreds({ ...creds, username: e.target.value })} className={inputCls} />
                </div>
                <div className="relative group/i">
                  <Lock className="absolute left-4 top-3.5 w-5 h-5 text-gray-600 group-focus-within/i:text-white transition-colors" />
                  <input type="password" required autoComplete="current-password" placeholder="Password"
                    value={creds.password} onChange={e => setCreds({ ...creds, password: e.target.value })} className={inputCls} />
                </div>
                <button disabled={loading} type="submit" className="w-full bg-white hover:bg-gray-100 text-black font-bold py-3.5 rounded-xl transition-all flex items-center justify-center mt-2">
                  {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Sign In'}
                </button>
              </motion.form>
            )}

            {/* STEP 1 */}
            {!isLogin && step === 1 && (
              <motion.div key="s1" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} transition={{ duration: 0.25 }}>
                <div className="flex bg-white/5 p-1 rounded-xl mb-5 border border-white/10">
                  {(['candidate', 'recruiter'] as const).map(r => (
                    <button key={r} type="button" onClick={() => setRole(r)}
                      className={`flex-1 py-2.5 text-sm font-semibold rounded-lg transition-all ${role === r ? 'bg-white/15 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
                      {r === 'candidate' ? 'Applicant' : 'Recruiter'}
                    </button>
                  ))}
                </div>
                <div className="space-y-4">
                  <div className="relative group/i">
                    <User className="absolute left-4 top-3.5 w-5 h-5 text-gray-600 group-focus-within/i:text-white transition-colors" />
                    <input type="text" required placeholder="Choose a username" autoComplete="username"
                      value={creds.username} onChange={e => setCreds({ ...creds, username: e.target.value })} className={inputCls} />
                  </div>
                  <div className="relative group/i">
                    <Mail className="absolute left-4 top-3.5 w-5 h-5 text-gray-600 group-focus-within/i:text-white transition-colors" />
                    <input type="email" required placeholder="Email address" autoComplete="email"
                      value={creds.email} onChange={e => setCreds({ ...creds, email: e.target.value })} className={inputCls} />
                  </div>
                  <div className="relative group/i">
                    <Lock className="absolute left-4 top-3.5 w-5 h-5 text-gray-600 group-focus-within/i:text-white transition-colors" />
                    <input type="password" required placeholder="Password (min. 8 characters)" autoComplete="new-password"
                      value={creds.password} onChange={e => setCreds({ ...creds, password: e.target.value })} className={inputCls} />
                  </div>
                  <button type="button" onClick={goToStep2}
                    className="w-full bg-white hover:bg-gray-100 text-black font-bold py-3.5 rounded-xl transition-all flex items-center justify-center gap-2">
                    Continue <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </motion.div>
            )}

            {/* STEP 2 */}
            {!isLogin && step === 2 && (
              <motion.div key="s2" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} transition={{ duration: 0.25 }}>
                <div className="flex bg-white/5 p-1 rounded-xl mb-5 border border-white/10">
                  {(['upload', 'manual'] as const).map(m => (
                    <button key={m} type="button" onClick={() => setEntryMode(m)}
                      className={`flex-1 py-2.5 text-xs font-bold uppercase tracking-widest rounded-lg transition-colors flex items-center justify-center gap-2 ${entryMode === m ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'}`}>
                      {m === 'upload' ? <><UploadCloud className="w-3.5 h-3.5" />Upload PDF</> : <><Keyboard className="w-3.5 h-3.5" />Fill Manually</>}
                    </button>
                  ))}
                </div>
                {entryMode === 'upload' ? (
                  <div>
                    <div onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
                      onDrop={e => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) setFile(f); }}
                      className={`border-2 border-dashed rounded-2xl p-10 flex flex-col items-center text-center transition-all cursor-pointer ${dragging ? 'border-white/50 bg-white/5' : 'border-white/10 hover:border-white/25 hover:bg-white/[0.02]'}`}
                      onClick={() => document.getElementById('pdf-upload')?.click()}>
                      <UploadCloud className={`w-9 h-9 mb-3 transition-colors ${dragging ? 'text-white' : 'text-gray-600'}`} />
                      <p className="text-white font-semibold mb-1">{role === 'candidate' ? 'Drop your CV / Resume here' : 'Drop your Job Description PDF'}</p>
                      <p className="text-gray-500 text-xs">PDF, DOC, DOCX — we'll extract your details automatically</p>
                      <input id="pdf-upload" type="file" className="hidden" accept=".pdf,.doc,.docx" onChange={e => setFile(e.target.files?.[0] || null)} />
                    </div>
                    {file && (
                      <div className="mt-3 flex items-center justify-between bg-white/5 border border-white/10 rounded-xl px-4 py-2.5">
                        <div className="flex items-center gap-2"><FileText className="w-4 h-4 text-gray-400" /><span className="text-sm text-white truncate max-w-[220px]">{file.name}</span></div>
                        <button type="button" onClick={() => setFile(null)} className="text-gray-500 hover:text-white ml-2"><X className="w-3.5 h-3.5" /></button>
                      </div>
                    )}
                    <button disabled={!file || parsing} onClick={handleParse}
                      className="w-full mt-4 bg-white hover:bg-gray-100 text-black font-bold py-3.5 rounded-xl transition-all disabled:opacity-40 flex items-center justify-center gap-2">
                      {parsing ? <><Loader2 className="w-4 h-4 animate-spin" />Extracting...</> : 'Extract & Continue'}
                    </button>
                    <p className="text-center text-xs text-gray-600 mt-3">Missing info from PDF? You'll complete it on the next step.</p>
                  </div>
                ) : (
                  role === 'candidate'
                    ? <CandidateForm data={cand} onChange={setCand} extraction={null} loading={loading} onSubmit={handleSignup} />
                    : <RecruiterForm data={job} onChange={setJob} extraction={null} loading={loading} onSubmit={handleSignup} />
                )}
              </motion.div>
            )}

            {/* STEP 3 — confirm extracted */}
            {!isLogin && step === 3 && (
              <motion.div key="s3" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} transition={{ duration: 0.25 }}>
                {role === 'candidate'
                  ? <CandidateForm data={cand} onChange={setCand} extraction={extraction} loading={loading} onSubmit={handleSignup} />
                  : <RecruiterForm data={job} onChange={setJob} extraction={extraction} loading={loading} onSubmit={handleSignup} />
                }
              </motion.div>
            )}
          </AnimatePresence>

          <div className="mt-7 pt-5 border-t border-white/5 text-center">
            <p className="text-sm text-gray-500">
              {isLogin ? "Don't have an account? " : 'Already have an account? '}
              <button onClick={reset} className="text-white hover:text-gray-300 font-semibold transition-colors underline underline-offset-4 decoration-white/20">
                {isLogin ? 'Sign up free' : 'Sign in'}
              </button>
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
