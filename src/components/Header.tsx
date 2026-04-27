import { Search, Bell, HelpCircle, User, Moon, Sun, PanelLeftClose, PanelLeft, Languages } from 'lucide-react';
import { useLang } from '../contexts/useLang';
import './Header.css';

interface HeaderProps {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  toggleSidebar: () => void;
  sidebarOpen?: boolean;
}

export default function Header({ theme, toggleTheme, toggleSidebar, sidebarOpen }: HeaderProps) {
  const { t, lang, toggleLang } = useLang();

  return (
    <header className="header">
      <div className="header-left">
        <button
          className="icon-btn menu-btn"
          onClick={toggleSidebar}
          title={t.header.toggleSidebar}
        >
          {sidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeft size={20} />}
        </button>
        <div className="search-bar">
          <Search size={16} className="search-icon" />
          <input type="text" placeholder={t.header.search} />
          <div className="search-shortcut">⌘K</div>
        </div>
      </div>

      <div className="header-right">
        {/* Language Toggle */}
        <button
          className="lang-toggle-btn"
          onClick={toggleLang}
          title={t.header.language}
        >
          <Languages size={15} />
          <span className="lang-label">{lang.toUpperCase()}</span>
        </button>

        <button className="icon-btn" onClick={toggleTheme} title={t.header.toggleTheme}>
          {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
        </button>
        <div className="divider" />
        <button className="icon-btn">
          <HelpCircle size={18} />
        </button>
        <button className="icon-btn has-badge">
          <Bell size={18} />
          <span className="badge" />
        </button>
        <button className="user-profile-btn">
          <div className="user-avatar">
            <User size={16} />
          </div>
        </button>
      </div>
    </header>
  );
}
