/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#10a37f',
        'primary-hover': '#0e8e70',
        'primary-light': '#e6f7f4',
        'sidebar-bg': '#202123',
        'sidebar-hover': '#2d2d3f',
        'chat-bg': '#f7f7f8',
        'message-user-bg': '#dcf8f6',
        'message-ai-bg': '#ffffff',
        'border-color': '#e5e7eb',
        'text-primary': '#111827',
        'text-secondary': '#6b7280',
        'text-light': '#f9fafb',
      },
      boxShadow: {
        sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
      },
      borderRadius: {
        sm: '0.375rem',
        md: '0.5rem',
        lg: '0.75rem',
      },
    },
  },
  plugins: [],
}; 