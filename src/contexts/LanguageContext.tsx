import { useState, useCallback } from 'react';
import { translations, type Lang, type Translations } from '../i18n';
import { LanguageContext } from './languageContextCore';

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>(() => {
    return (localStorage.getItem('lang') as Lang) || 'tr';
  });

  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    localStorage.setItem('lang', l);
  }, []);

  const toggleLang = useCallback(() => {
    setLang(lang === 'tr' ? 'en' : 'tr');
  }, [lang, setLang]);

  const t = translations[lang] as Translations;

  return (
    <LanguageContext.Provider value={{ lang, t, setLang, toggleLang }}>
      {children}
    </LanguageContext.Provider>
  );
}
