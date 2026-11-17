# WSDM Cup 2027

Website for the WSDM Cup 2027 challenges at ACM WSDM 2027
(February 15–19, 2027, Hong Kong).

| URL | Serves |
|---|---|
| <https://wsdm2027-cup.github.io/> | overview page listing the challenges |
| <https://wsdm2027-cup.github.io/agenticsearch/> | *Agentic Search over Instruction Trajectories* challenge site |

Repository: `wsdm2027-cup/wsdm2027-cup.github.io`. The repository name must match
`<account>.github.io` exactly, otherwise GitHub publishes the site under a
sub-path and the domain root stays unavailable. Because the root is claimed by
this repository, each challenge lives in a sub-directory — a plain folder
`agenticsearch/` is already reachable at `/agenticsearch/`, with no extra
repository or configuration needed.

## Structure

```
index.html                  overview page, links to each challenge
agenticsearch/
  index.html                challenge content
  styles.css                layout and typography
  site.js                   mobile navigation and scroll highlighting
  assets/                   locally hosted Montserrat fonts (licences included)
.nojekyll                   serve files as-is, skip Jekyll processing
```

All asset references are relative, so a challenge folder can be renamed, or
moved to the domain root, without editing any HTML or CSS.

No build step, no backend, no runtime third-party requests. Preview over HTTP:

```bash
python -m http.server 8766 --bind 127.0.0.1
```

Then open <http://127.0.0.1:8766/> for the overview and
<http://127.0.0.1:8766/agenticsearch/> for the challenge.

## Publishing

GitHub Pages is served from the `main` branch root:

1. Push `main`.
2. Settings → Pages → Source: *Deploy from a branch*, branch `main`, folder `/ (root)`.
3. The site rebuilds roughly a minute after each push.

## Adding a challenge

1. Create a folder named after the URL segment you want, for example `communitynotes/`.
2. Put its `index.html` inside and keep all references relative.
3. Add an `<a class="card">` block in the root `index.html` pointing at `communitynotes/`.

## Custom domain

Add a `CNAME` file containing the bare hostname, create a `CNAME` DNS record
pointing at `wsdm2027-cup.github.io`, then enable *Enforce HTTPS*. Paths keep
working unchanged.

## Scope

This repository holds the public website only. Challenge data, private evaluation
tasks, ground-truth trajectories and generation tooling are kept out of it.
