# WSDM Cup 2027 — Agentic Search

Official website for the *Agentic Search over Instruction Trajectories* challenge at
ACM WSDM 2027 (February 15–19, 2027, Hong Kong).

Live at <https://wsdmcup2027-agenticsearch.github.io>.

## Structure

```
index.html    page content
styles.css    layout and typography
site.js       mobile navigation and scroll highlighting
assets/       locally hosted Montserrat fonts (licences included)
.nojekyll     serve files as-is, skip Jekyll processing
```

No build step, no backend, no runtime third-party requests. Open `index.html`
directly in a browser, or preview over HTTP:

```bash
python -m http.server 8766 --bind 127.0.0.1
```

All asset references are relative, so the site works at a domain root and at a
sub-path without changes.

## Publishing

GitHub Pages is served from the `main` branch root:

1. Push `main`.
2. Settings → Pages → Source: *Deploy from a branch*, branch `main`, folder `/ (root)`.
3. The site rebuilds roughly a minute after each push.

## Custom domain

Add a `CNAME` file containing the bare hostname (for example
`agentic-search.wsdmcup2027.org`), create a `CNAME` DNS record pointing at
`wsdmcup2027-agenticsearch.github.io`, then enable *Enforce HTTPS*.

## Scope

This repository holds the public website only. Challenge data, private evaluation
tasks, ground-truth trajectories and generation tooling are kept out of it.
