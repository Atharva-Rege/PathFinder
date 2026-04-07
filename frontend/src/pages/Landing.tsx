import { motion } from 'framer-motion';
import { ArrowRight, BrainCircuit } from 'lucide-react';
import { Link } from 'react-router-dom';



export default function Landing() {
  return (
    <div className="relative min-h-[calc(100vh-80px)] flex flex-col items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="z-10 text-center max-w-4xl px-4"
      >
        <div className="inline-flex items-center justify-center p-3 mb-6 rounded-full glass border border-white/20">
          <BrainCircuit className="text-gray-300 w-5 h-5 mr-2" />
          <span className="text-gray-300 font-medium tracking-wide text-sm uppercase">GNN-Powered Job Search</span>
        </div>

        <h1 className="text-5xl md:text-7xl font-bold mb-6 tracking-tight">
          Find Your Path with <span className="bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-500">Pathfinder</span>
        </h1>

        <p className="text-lg md:text-xl text-gray-400 mb-12 max-w-2xl mx-auto leading-relaxed">
          The job search engine utilizing Temporal Graph Neural Networks to semantically match candidates with their ideal roles in real-time.
        </p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-12"
        >
          <Link to="/auth" className="inline-flex items-center space-x-2 bg-white text-black px-8 py-4 rounded-xl font-bold transition-all duration-300 hover:scale-105 hover:bg-gray-200 shadow-[0_0_40px_rgba(255,255,255,0.2)]">
            <span>Get Started — It's Free</span>
            <ArrowRight className="w-5 h-5" />
          </Link>
        </motion.div>
      </motion.div>
    </div>
  );
}
