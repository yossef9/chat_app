import { useState, useEffect, useRef } from 'react';

const ResizableLayout = ({ children }) => {
  const [leftWidth, setLeftWidth] = useState(320); // Sidebar width
  const [rightWidth, setRightWidth] = useState(400); // Document section width
  const [isLeftResizing, setIsLeftResizing] = useState(false);
  const [isRightResizing, setIsRightResizing] = useState(false);
  const [isRightVisible, setIsRightVisible] = useState(true);
  
  const containerRef = useRef(null);
  const leftResizeRef = useRef(null);
  const rightResizeRef = useRef(null);

  // Load saved preferences from localStorage
  useEffect(() => {
    const savedLeftWidth = localStorage.getItem('chat_left_width');
    const savedRightWidth = localStorage.getItem('chat_right_width');
    const savedRightVisible = localStorage.getItem('chat_right_visible');
    
    if (savedLeftWidth) setLeftWidth(parseInt(savedLeftWidth));
    if (savedRightWidth) setRightWidth(parseInt(savedRightWidth));
    if (savedRightVisible !== null) setIsRightVisible(savedRightVisible === 'true');
  }, []);

  // Save preferences to localStorage
  useEffect(() => {
    localStorage.setItem('chat_left_width', leftWidth.toString());
  }, [leftWidth]);

  useEffect(() => {
    localStorage.setItem('chat_right_width', rightWidth.toString());
  }, [rightWidth]);

  useEffect(() => {
    localStorage.setItem('chat_right_visible', isRightVisible.toString());
  }, [isRightVisible]);

  const handleMouseDown = (e, type) => {
    e.preventDefault();
    if (type === 'left') {
      setIsLeftResizing(true);
    } else if (type === 'right') {
      setIsRightResizing(true);
    }
  };

  const handleMouseMove = (e) => {
    if (!containerRef.current) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const containerWidth = containerRect.width;

    if (isLeftResizing) {
      const newLeftWidth = e.clientX - containerRect.left;
      const minWidth = 200;
      const maxWidth = isRightVisible ? containerWidth - rightWidth - 100 : containerWidth - 100;
      
      if (newLeftWidth >= minWidth && newLeftWidth <= maxWidth) {
        setLeftWidth(newLeftWidth);
      }
    }

    if (isRightResizing) {
      const newRightWidth = containerRect.right - e.clientX;
      const minWidth = 200;
      const maxWidth = containerWidth - leftWidth - 100;
      
      if (newRightWidth >= minWidth && newRightWidth <= maxWidth) {
        setRightWidth(newRightWidth);
      }
    }
  };

  const handleMouseUp = () => {
    setIsLeftResizing(false);
    setIsRightResizing(false);
  };

  useEffect(() => {
    if (isLeftResizing || isRightResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };
    }
  }, [isLeftResizing, isRightResizing]);

  const toggleRightSection = () => {
    setIsRightVisible(!isRightVisible);
  };

  return (
    <div ref={containerRef} className="flex h-full w-full">
      {/* Left Sidebar */}
      <div 
        className="bg-gray-50 border-r border-gray-200 flex-shrink-0"
        style={{ width: `${leftWidth}px` }}
      >
        {children[0]}
      </div>

      {/* Left Resize Handle */}
      <div
        ref={leftResizeRef}
        className="w-1 bg-gray-300 hover:bg-blue-500 cursor-col-resize flex-shrink-0 transition-colors duration-200"
        onMouseDown={(e) => handleMouseDown(e, 'left')}
      />

      {/* Main Chat Area */}
      <div className="flex-1 min-w-0 h-full">
        {children[1]}
      </div>

      {/* Right Resize Handle */}
      {isRightVisible && (
        <div
          ref={rightResizeRef}
          className="w-1 bg-gray-300 hover:bg-blue-500 cursor-col-resize flex-shrink-0 transition-colors duration-200"
          onMouseDown={(e) => handleMouseDown(e, 'right')}
        />
      )}

      {/* Right Document Section */}
      {isRightVisible && (
        <div 
          className="bg-gray-50 border-l border-gray-200 flex-shrink-0"
          style={{ width: `${rightWidth}px` }}
        >
          {children[2]}
        </div>
      )}

      {/* Toggle Button for Right Section */}
      <button
        onClick={toggleRightSection}
        className={`fixed top-1/2 transform -translate-y-1/2 z-20 w-6 h-12 flex items-center justify-center rounded-r-lg transition-all duration-300 shadow-lg ${
          isRightVisible 
            ? 'right-0 bg-gray-200 hover:bg-gray-300 text-gray-600 border border-l-0 border-gray-300' 
            : 'right-0 bg-blue-500 hover:bg-blue-600 text-white'
        }`}
        title={isRightVisible ? 'Hide Documents' : 'Show Documents'}
      >
        <svg 
          className={`w-3 h-3 transition-transform duration-300 ${isRightVisible ? 'rotate-180' : ''}`} 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>
    </div>
  );
};

export default ResizableLayout;
