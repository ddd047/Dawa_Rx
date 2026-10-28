"use client";

import { useState, useCallback, useRef } from "react";

interface Props {
  onFileSelected: (file: File) => void;
  isLoading: boolean;
}

export function UploadCard({ onFileSelected, isLoading }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(file);
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div
      className={`upload-card ${isDragging ? "upload-card--dragging" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
      onClick={() => !isLoading && fileInputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="Upload prescription image"
      onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
    >
      <input
        ref={fileInputRef}
        id="prescription-upload"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        className="sr-only"
        onChange={onInputChange}
        disabled={isLoading}
      />

      {preview ? (
        <div className="upload-card__preview">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={preview} alt="Prescription preview" className="upload-card__image" />
          {!isLoading && (
            <p className="upload-card__replace">Click or drop to replace</p>
          )}
        </div>
      ) : (
        <div className="upload-card__empty">
          <div className="upload-card__icon">📋</div>
          <p className="upload-card__title">Upload Prescription</p>
          <p className="upload-card__subtitle">Drag & drop or click to browse</p>
          <p className="upload-card__hint">JPG, PNG or WebP · Max 8 MB</p>
        </div>
      )}

      {isLoading && (
        <div className="upload-card__overlay">
          <div className="spinner" aria-label="Analysing prescription" />
        </div>
      )}
    </div>
  );
}
