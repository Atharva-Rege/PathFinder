import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import NetworkBackground from './components/NetworkBackground';
import Landing from './pages/Landing';
import CandidatePortal from './pages/CandidatePortal';
import RecruiterPortal from './pages/RecruiterPortal';
import Auth from './pages/Auth';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background text-white relative overflow-hidden">
        <NetworkBackground />
        <Navbar />
        <div className="relative z-10 pt-20">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/auth" element={<Auth />} />
            <Route path="/candidate/*" element={<CandidatePortal />} />
            <Route path="/recruiter/*" element={<RecruiterPortal />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
