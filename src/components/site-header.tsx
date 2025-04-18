'use client';

import Link from 'next/link';

export default function SiteHeader() {
  // External links data
  const navLinks = [
    { href: 'https://breakout.maikers.com', label: 'MAIKERS BREAKOUT' },
    { href: 'https://baxus.co', label: 'Baxus' },
    {
      href: 'https://github.com/chainsona/baxus-ai-agent-bob',
      label: 'GitHub',
    },
  ];

  return (
    <header className="border-b border-border/20 bg-background backdrop-blur-md sticky top-0 z-50">
      <div className="container flex h-16 items-center justify-between py-2">
        <div className="flex items-center gap-2">
          <Link href="/" className="flex items-center">
            <span className="text-xl font-bold tracking-tight text-foreground">
              B A <span className="text-primary">X</span> U S
            </span>
            <span className="ml-2 px-2 py-1 bg-primary/10 border border-primary/20 rounded text-xs font-medium text-primary">
              BOB
            </span>
          </Link>
        </div>
        <div className="flex items-center gap-6">
          <nav className="hidden md:flex items-center space-x-8 text-sm font-medium">
            {navLinks.map((link) => (
              <Link
                key={link.label}
                href={link.href}
                className="transition-colors hover:text-primary"
                target="_blank"
                rel="noopener noreferrer"
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
      </div>
    </header>
  );
}
