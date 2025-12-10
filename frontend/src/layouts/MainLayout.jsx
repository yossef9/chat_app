import { Outlet } from "react-router";
import Sidebar from "../components/Sidebar";
import UserProfile from "../components/UserProfile";
import ResizableLayout from "../components/ResizableLayout";
import DocumentView from "../components/DocumentView";

const MainLayout = () => {
  return (
    <div className="h-screen overflow-hidden bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-900">Document Chat AI</h1>
            <p className="text-sm text-gray-500">Chat with your documents using AI</p>
          </div>
          <UserProfile />
        </div>
      </header>
      
      {/* Main Content with Resizable Layout */}
      <div className="h-[calc(100vh-80px)] overflow-hidden">
        <ResizableLayout>
          <Sidebar />
          <div className="flex-1 h-full">
            <Outlet />
          </div>
          <DocumentView />
        </ResizableLayout>
      </div>
    </div>
  );
};

export default MainLayout;
