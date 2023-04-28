import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import './App.css';
import HomePage from './components/Home';
import SearchResultsPage from './components/SearchResultsPage';

const router = createBrowserRouter([
  {
    path: '/',
    element: <HomePage />,
  },
  {
    path: '/search',
    element: <SearchResultsPage />,
  },
]);

function App() {
  return <RouterProvider router={router} />;
}

export default App;
