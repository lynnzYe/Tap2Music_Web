
import React, { useEffect, useRef } from 'react';
import { NoteEvent, PIANO_CONFIG, getKeyLayout } from '../types';

interface VisualizerProps {
  noteEvents: NoteEvent[];
  activeNotes: Set<number>;
}

const Visualizer: React.FC<VisualizerProps> = ({ noteEvents, activeNotes }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;

    const render = () => {
      const now = performance.now();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw subtle pitch lanes to match the keyboard layout
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
      ctx.lineWidth = 1;
      
      // We only draw lanes for white key boundaries for a cleaner look
      const whiteWidth = canvas.width / PIANO_CONFIG.NUM_WHITE_KEYS;
      for (let i = 0; i <= PIANO_CONFIG.NUM_WHITE_KEYS; i++) {
        const x = i * whiteWidth;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }

      // Draw note blocks
      noteEvents.forEach(note => {
        const layout = getKeyLayout(note.pitch);
        const x = (layout.x / 100) * canvas.width;
        const width = (layout.width / 100) * canvas.width;
        
        const startY = canvas.height - (now - note.startTime) * PIANO_CONFIG.SCROLL_SPEED;
        const endTime = note.endTime || now;
        const endY = canvas.height - (now - endTime) * PIANO_CONFIG.SCROLL_SPEED;
        
        const rectTop = Math.min(startY, endY);
        const rectBottom = Math.max(startY, endY);
        const rectHeight = Math.max(5, rectBottom - rectTop);
        
        if (rectBottom > 0 && rectTop < canvas.height) {
          const isActive = activeNotes.has(note.pitch) && !note.endTime;
          
          ctx.fillStyle = note.color;
          
          if (isActive) {
            ctx.shadowBlur = 15;
            ctx.shadowColor = note.color;
          } else {
            ctx.shadowBlur = 0;
            ctx.globalAlpha = 0.6;
          }
          
          const radius = Math.max(0, Math.min(4, rectHeight / 2, width / 2));
          
          ctx.beginPath();
          ctx.roundRect(x + 1, rectTop, width - 2, rectHeight, radius);
          ctx.fill();
          
          ctx.globalAlpha = 1.0;
          ctx.shadowBlur = 0;
        }
      });

      animationFrameId = requestAnimationFrame(render);
    };

    const handleResize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };

    window.addEventListener('resize', handleResize);
    handleResize();
    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
    };
  }, [noteEvents, activeNotes]);

  return (
    <div className="flex-grow w-full relative bg-slate-950 overflow-hidden">
      <canvas 
        ref={canvasRef} 
        className="w-full h-full"
      />
      {/* Bottom connection glow */}
      <div className="absolute bottom-0 left-0 right-0 h-1 bg-indigo-500/30 blur-md pointer-events-none" />
    </div>
  );
};

export default Visualizer;
