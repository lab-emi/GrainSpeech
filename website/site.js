(() => {
  const players = [...document.querySelectorAll("audio")];
  const buttons = [...document.querySelectorAll("[data-sample]")];
  const panels = [...document.querySelectorAll(".sample-panel")];

  // Keep speech intelligible: starting one player pauses every other player.
  document.addEventListener(
    "play",
    (event) => {
      if (!(event.target instanceof HTMLAudioElement)) return;
      players.forEach((player) => {
        if (player !== event.target) player.pause();
      });
    },
    true,
  );

  buttons.forEach((button) =>
    button.addEventListener("click", () => {
      const active = panels.find((panel) => !panel.hidden);
      const keepComparisonsOpen =
        active?.querySelector("details").open ?? false;
      const selected = button.dataset.sample;
      buttons.forEach((item) =>
        item.setAttribute("aria-pressed", String(item === button)),
      );
      panels.forEach((panel) => {
        panel.hidden = panel.id !== selected;
        if (panel.hidden)
          panel.querySelectorAll("audio").forEach((player) => player.pause());
        else panel.querySelector("details").open = keepComparisonsOpen;
      });
    }),
  );

  const featured = document.querySelector("#featured-audio");
  ["play", "pause", "ended"].forEach((type) =>
    featured.addEventListener(type, () => {
      featured
        .closest(".first-listen")
        .classList.toggle("is-playing", !featured.paused);
    }),
  );

  // Retry media errors with a same-origin Blob, including Safari range failures.
  // Retain a visible download link if the network itself is unavailable.
  players.forEach((player) => {
    const original = player.getAttribute("src");
    let retrying = false;
    let objectUrl;
    let notice;
    const showNotice = (message) => {
      if (!notice) {
        notice = document.createElement("p");
        notice.className = "audio-error";
        notice.setAttribute("role", "status");
        player.after(notice);
      }
      notice.textContent = message + " ";
      const link = document.createElement("a");
      link.href = original;
      link.textContent = "Open the WAV file";
      notice.append(link);
    };
    player.addEventListener("playing", () => {
      if (notice) notice.remove();
      notice = null;
    });
    player.addEventListener("error", async () => {
      if (retrying) {
        showNotice("Unable to play here.");
        return;
      }
      retrying = true;
      try {
        const response = await fetch(original);
        if (!response.ok) throw new Error("Audio request failed");
        const bytes = await response.arrayBuffer();
        objectUrl = URL.createObjectURL(
          new Blob([bytes], { type: "audio/wav" }),
        );
        player.src = objectUrl;
        player.load();
        // A completed retry must not restart speech after the visitor switches
        // sentences, pauses, or starts another recording.
        showNotice("Audio reloaded. Press play to listen.");
      } catch {
        showNotice("Unable to play here.");
      }
    });
    window.addEventListener("pagehide", (event) => {
      if (objectUrl && !event.persisted) URL.revokeObjectURL(objectUrl);
    });
  });

  const copyButton = document.querySelector("#copy-citation");
  const copyStatus = document.querySelector("#copy-status");
  if (navigator.clipboard?.writeText) {
    copyButton.hidden = false;
    copyButton.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(
          document.querySelector("#bibtex").textContent,
        );
        copyButton.textContent = "Copied";
        copyStatus.textContent = "BibTeX copied to clipboard.";
      } catch {
        copyStatus.textContent =
          "Copy was unavailable. Select and copy the citation below.";
        copyButton.textContent = "Select text below";
      }
    });
  }
})();
