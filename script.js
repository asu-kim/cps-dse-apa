const API_BASE = window.location.hostname.endsWith('github.io')
 ? 'https://cpsdseprecisionagriculture-precision-agriculture-backend.hf.space'
  : 'http://127.0.0.1:8000';
document.addEventListener("DOMContentLoaded", () => {
  console.log("DOM Fully Loaded");
});
let latestResult = null;

async function fetchOptions() {
  const budgetInput   = document.getElementById("budget").value;
  const farmSizeInput = document.getElementById("farm_size").value;
  const cropTypeInput = document.getElementById("crop_type").value;
  const evalMode      = document.getElementById("eval_mode").value;

  const applications = Array.from(document.querySelectorAll('#applications input[type="checkbox"]:checked'))
    .map(cb => cb.value);

  const requestData = {
    budget: parseFloat(budgetInput),
    farm_size: parseFloat(farmSizeInput),
    crop_type: cropTypeInput,
    applications
  };

  if (isNaN(requestData.budget) || isNaN(requestData.farm_size) || !requestData.crop_type) {
    alert("Please fill out all fields correctly before submitting.");
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/get_options`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const rawResult = await response.json();
    latestResult = { inputs: requestData, ...rawResult };
    console.log(" Response Received:", rawResult);

    if (evalMode === "self") {
      displayOptions({
        proposed_approach: rawResult.proposed_approach,
        area_payload: rawResult.area_payload,
        cost_area: rawResult.cost_area,
        payload_cost: rawResult.payload_cost
      });
    } else if (evalMode === "optimizer") {
      displayOptions({
        proposed_approach: rawResult.proposed_approach,
        simulated_annealing: rawResult.simulated_annealing,
        bayesian: rawResult.bayesian,
        random_search: rawResult.random_search,
        genetic_algorithm: rawResult.genetic_algorithm,
        pg_dse: rawResult.pg_dse,
        discrete: rawResult.discrete,
        lengler: rawResult.lengler,
        portfolio: rawResult.portfolio
      });
    }
  } catch (error) {
    console.error(" Error Fetching Options:", error);
    document.getElementById("options").innerHTML = `<p>Error fetching data. Please try again.</p>`;
  }
}

function displayOptions(data) {
  const optionsDiv = document.getElementById("options");
  optionsDiv.innerHTML = ""; // Clear previous results

  let html = `<div class="results-flex">`;

  for (const [key, configsArray] of Object.entries(data)) {
    let title = {
      proposed_approach: "Proposed Approach",
      area_payload: "Area + Payload",
      cost_area: "Cost + Area",
      payload_cost: "Payload + Cost",
      simulated_annealing: "Simulated Annealing",
      bayesian: "bayesian",
      random_search: "Random Search",
      genetic_algorithm: "Genetic Algorithm",
      pg_dse: "PG-DSE",
      discrete: "discrete",
      lengler: "lengler",
      portfolio: "portfolio"
    }[key] || key;

    let colHtml = `<div class="column"><h2>${title}</h2>`;

    if (!configsArray || configsArray.length === 0) {
      colHtml += `<p><em>No feasible config</em></p>`;
    } else {
      configsArray.forEach((config, idx) => {
        colHtml += `
          <div class="config-card">
            <h3>Configuration ${idx + 1} — Quantity: ${config.quantity} </h3>
            <p>Type: ${config.type}</p>
            <p>Body: ${config.body}</p>
            <p>Motor: ${config.motor}</p>
            <p>Battery: ${config.battery}</p>
            <p>Computing Device (On-device): ${config.computing_device}</p>
            <p>Computing Type: ${config.computing_type}</p>
            <p>  - Cost: $${(config.computing_cost ?? 0).toFixed(2)}</p>
            <p>  - Performance Score: ${config.computing_perf ?? "N/A"}</p>
            <p>  - Power: ${(config.computing_power_watts ?? 0).toFixed(2)} W</p>
            <p>Edge Server: ${config.edge_server?.name ?? "N/A"} 
                ($${(config.edge_server?.cost ?? 0).toFixed(2)})</p>
            <p>Base Cost: $${(config.base_cost ?? 0).toFixed(2)}</p>
            <p>Additional Components: ${Array.isArray(config.additional_components)
              ? config.additional_components.map(c => `${c.name} (${c.category})`).join(", ")
              : "None"}</p>
            <p>Additional Cost: $${(config.additional_cost ?? 0).toFixed(2)}</p>
            <p>Total Cost: $${(config.total_cost ?? 0).toFixed(2)}</p>
            <p>Coverage: ${(config.coverage ?? 0).toFixed(2)} m²</p>
            <p>Payload Capacity: ${(config.payload ?? 0).toFixed(2)} kg</p>
            <p>Estimated Runtime: ${(config.runtime_hours ?? 0).toFixed(2)} hours</p>
          </div>
          <hr>
        `;
      });
    }

    colHtml += `</div>`; // close .column
    html += colHtml;
  }

  html += `</div>`; // close .results-flex
  optionsDiv.innerHTML = html;
  console.log("HTML updated successfully!");
}

function downloadJSON() {
  if (!latestResult) {
    alert("No data to download. Please run the optimizer first.");
    return;
  }

  const blob = new Blob([JSON.stringify(latestResult, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'results.json';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
