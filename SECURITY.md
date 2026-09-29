<!-- Modified by meowkernels. -->
# Security

Please report vulnerabilities privately to the maintainers of
[github.com/abhishekgahlot2/fastkernel](https://github.com/abhishekgahlot2/fastkernel) (a GitHub security advisory),
not in a public issue. Include the commit, the Mac and macOS version, and steps to reproduce.

The server listens on `127.0.0.1`, and authentication is off by default. Set `SPLASH_API_KEY` before exposing the
server beyond the local machine. Exposing it without a key or a proxy is outside the threat model.

fastkernel is built on Splash; the server and this threat model come from Splash.
