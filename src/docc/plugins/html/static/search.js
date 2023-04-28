window.initialize_search = function(search_path, search_base) {
    "use strict";

    window.addEventListener("DOMContentLoaded", () => {
        const searchElement = document.getElementById("search-results");

        const tag = document.createElement("script");
        tag.async = true;
        tag.src = search_path;

        const searcherPromise = new Promise((resolve, reject) => {
            const onLoad = () => {
                tag.removeEventListener("load", onLoad);
                tag.removeEventListener("error", onError);

                const options = {
                    keys: [
                        "content.name",
                        "content.text"
                    ]
                };

                resolve(new Fuse(window.SEARCH_INDEX, options));
            };

            const onError = (e) => {
                tag.removeEventListener("load", onLoad);
                tag.removeEventListener("error", onError);
                reject(e);
            };

            tag.addEventListener("load", onLoad);
            tag.addEventListener("error", onError);

            document.body.appendChild(tag);
        });

        const bars = document.querySelectorAll(".search-bar input[type='search']");

        const onTimeout = async (e) => {
            delete e.target.dataset.timeoutId;
            if (!e.target.value) {
                searchElement.replaceChildren();
                return;
            }

            const searcher = await searcherPromise;
            const results = searcher.search(e.target.value);

            searchElement.replaceChildren(...results.map((r) => {
                const href = new URL(
                    r.item.source.path + ".html",
                    new URL(search_base, window.location)
                );

                if (r.item.source.identifier) {
                    const specifier = r.item.source.specifier || 0;
                    href.hash = `#${r.item.source.identifier}:${specifier}`;
                }

                const anchor = document.createElement("a");
                anchor.innerText = r.item.content.name;
                anchor.href = href;

                const path = document.createElement("span");
                path.classList.add("search-path");
                path.innerText = " " + r.item.source.path

                const elem = document.createElement("li");
                elem.appendChild(anchor);
                elem.appendChild(path);

                if (r.item.content.text) {
                    const text = document.createElement("div");
                    text.classList.add("search-text");
                    text.innerText = " " + r.item.content.text;
                    elem.appendChild(text);
                }

                return elem;
            }));
        };

        const onInput = async (e) => {
            const timeoutIdText = e.target.dataset.timeoutId;
            if (timeoutIdText !== undefined) {
                const timeoutId = Number.parseInt(timeoutIdText);
                clearTimeout(timeoutId);
            }

            e.target.dataset.timeoutId = setTimeout(() => onTimeout(e), 300);
        };

        for (const bar of bars) {
            bar.addEventListener("input", onInput);
        }
    });
};
