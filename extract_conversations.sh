# ── Conversation Extraction ────────────────────────────────────────────────────
results_root="${results_root:-results-p}"
conv_output="conversations.md"
> "$conv_output"

echo "Extracting sample conversations..."

for dir in "$results_root"/*/; do
    [ -d "$dir" ] || continue
    dir="${dir%/}"
    benchmark_name=$(basename "$dir")

    conv_root="$dir/conversations/-downsample/gpt-4o"
    [ -d "$conv_root" ] || { echo "Warning: no conv dir for $benchmark_name — skipping" >&2; continue; }

    json_file=$(find "$conv_root" -name "*.json" ! -name "cost-downsample.json" 2>/dev/null | head -1)
    count=$(find "$conv_root" -name "*.json" ! -name "cost-downsample.json" 2>/dev/null | wc -l)
    [ -z "$json_file" ] && { echo "Warning: no scenario json in $conv_root — skipping" >&2; continue; }

    cost_file="$conv_root/cost-downsample.json"
    cost=$([ -f "$cost_file" ] && jq -r '.simulation.cost // "N/A"' "$cost_file" || echo "N/A")

    scenario_id=$(jq -r '.scenario_id // "unknown"' "$json_file")
    scenario_title=$(jq -r '.scenario.title // "untitled"' "$json_file")

    echo "## $benchmark_name" >> "$conv_output"
    echo "_Scenario: \`$scenario_id\` — ${scenario_title}_" >> "$conv_output"
    echo "_Cost: \$$(printf '%.4f' "$cost")_" >> "$conv_output"
    echo "_Note: There are a count of $count simulations._" >> "$conv_output"
    echo "" >> "$conv_output"

    jq -r '
      .samples[0][] |
      if .role == "user" then
        "**User:** " + .content + "\n"
      else
        "**Assistant:** " + .content + "\n"
      end
    ' "$json_file" >> "$conv_output"

    printf '–%.0s' {1..64} >> "$conv_output"
    printf '\n\n' >> "$conv_output"
done

echo "Conversations → $conv_output"