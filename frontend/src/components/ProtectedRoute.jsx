import { Navigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Loader2 } from 'lucide-react';
import './ProtectedRoute.css';

/**
 * ProtectedRoute Component
 *
 * Protects routes that require authentication
 * - Shows loading state while checking auth
 * - Redirects to login if not authenticated
 * - Renders children if authenticated
 */
export default function ProtectedRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="protected-loading">
        <Loader2 size={40} className="spinner" />
        <p>載入中...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return children;
}
