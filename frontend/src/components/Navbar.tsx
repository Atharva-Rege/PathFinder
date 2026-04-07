import { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { BrainCircuit, LogOut } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const isLanding = location.pathname === '/';
  const isDashboard = location.pathname.startsWith('/candidate') || location.pathname.startsWith('/recruiter');
  const isAuthPage = location.pathname.startsWith('/auth');
  const blendNav = isLanding || isDashboard || isAuthPage;
  const [isAuthed, setIsAuthed] = useState(false);
  const [role, setRole] = useState<string | null>(null);

  // Re-check token on every route change
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      setIsAuthed(true);
      // Determine role from current path or fetch from backend
      if (location.pathname.startsWith('/candidate')) setRole('candidate');
      else if (location.pathname.startsWith('/recruiter')) setRole('recruiter');
      else {
        // Ask backend for role
        fetch('http://127.0.0.1:8000/api/auth/me/', {
          headers: { Authorization: `Token ${token}` }
        }).then(r => r.ok ? r.json() : null).then(d => {
          if (d) setRole(d.role);
        }).catch(() => {});
      }
    } else {
      setIsAuthed(false);
      setRole(null);
    }
  }, [location.pathname]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsAuthed(false);
    setRole(null);
    navigate('/');
  };

  return (
    <nav className={`fixed top-0 w-full z-50 transition-all duration-300 ${blendNav ? 'bg-transparent' : 'bg-background/80 backdrop-blur-md border-b border-white/10'}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-20">
          <Link to={isAuthed ? `/${role || ''}` : '/'} className="flex items-center space-x-2">
            <BrainCircuit className="w-8 h-8 text-white" />
            <span className="font-bold text-xl tracking-wider text-white">PATHFINDER</span>
          </Link>

          <div className="flex items-center space-x-4">
            {isAuthed ? (
              <>
                <Link
                  to={`/${role}`}
                  className="text-gray-400 hover:text-white transition-colors text-sm font-medium"
                >
                  My Dashboard
                </Link>
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors font-medium"
                >
                  <LogOut className="w-4 h-4" />
                  Sign Out
                </button>
              </>
            ) : (
              <>
                <Link
                  to="/auth?mode=login"
                  className="text-gray-400 hover:text-white transition-colors text-sm font-medium"
                >
                  Sign In
                </Link>
                <Link
                  to="/auth?mode=signup"
                  className="bg-white text-black hover:bg-gray-100 px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-200"
                >
                  Get Started
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
