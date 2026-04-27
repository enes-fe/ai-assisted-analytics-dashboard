import { createContext } from 'react';
import type { Lang, Translations } from '../i18n';

export interface LanguageContextType {
  lang: Lang;
  t: Translations;
  setLang: (l: Lang) => void;
  toggleLang: () => void;
}

export const LanguageContext = createContext<LanguageContextType | undefined>(undefined);
