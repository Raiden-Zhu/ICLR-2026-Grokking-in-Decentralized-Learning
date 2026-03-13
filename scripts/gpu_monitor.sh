#!/bin/bash

# Configuration section: Modify here, the display below will automatically adapt
INT=10          # Sampling interval (seconds)
SAMPLES=100      # Number of samples (for sliding window)
WINDOW=$((INT * SAMPLES))  # Automatically calculate total window duration (seconds)

# History log files
HIST_FILE_GPU="/dev/shm/gpu_stats.log"
HIST_FILE_RAM="/dev/shm/ram_stats.log"

# Clean up old data
rm -f $HIST_FILE_GPU $HIST_FILE_RAM

# Dynamically displayed header variable
AVG_LABEL="Avg_${WINDOW}s"

echo "Initializing full system monitoring (${INT}s sampling, ${WINDOW}s window)..."

while true; do
    # ==========================
    # 1. Collect data
    # ==========================
    
    # GPU: index, memory.used, utilization.gpu
    CURRENT_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    
    # RAM: free -m (Used, Total)
    CURRENT_RAM=$(free -m | awk '/^Mem:/{print $3, $2}')

    if [ -z "$CURRENT_GPU" ]; then
        echo "Error: Unable to fetch GPU data."
        sleep 2
        continue
    fi

    # ==========================
    # 2. Save to history (Maintain sliding window)
    # ==========================

    # --- GPU ---
    echo "$CURRENT_GPU" >> $HIST_FILE_GPU
    GPU_COUNT=$(echo "$CURRENT_GPU" | wc -l)
    MAX_LINES_GPU=$((GPU_COUNT * SAMPLES))
    
    # Maintain line limit (High-performance method)
    if [ $(wc -l < "$HIST_FILE_GPU") -gt $MAX_LINES_GPU ]; then
        tail -n $MAX_LINES_GPU $HIST_FILE_GPU > "${HIST_FILE_GPU}.tmp" && mv "${HIST_FILE_GPU}.tmp" $HIST_FILE_GPU
    fi

    # --- RAM ---
    echo "$CURRENT_RAM" >> $HIST_FILE_RAM
    if [ $(wc -l < "$HIST_FILE_RAM") -gt $SAMPLES ]; then
        tail -n $SAMPLES $HIST_FILE_RAM > "${HIST_FILE_RAM}.tmp" && mv "${HIST_FILE_RAM}.tmp" $HIST_FILE_RAM
    fi

    # ==========================
    # 3. Render interface
    # ==========================
    clear
    
    echo "======================= SYSTEM MONITOR (Window: ${WINDOW}s) ======================="
    
    # --- 3.1 RAM (GB) ---
    awk '{
        used=$1; total=$2;
        sum+=used; count++;
        if(used > max) max=used;
        curr=used;
    } END {
        avg = (count > 0) ? sum/count : 0;
        printf "\033[1;33m[ System Memory ]\033[0m  Total: %.1f GB\n", total/1024
        printf "Usage (GB)    :  Curr: \033[1;32m%-6.1f\033[0m Avg: \033[1;33m%-6.1f\033[0m Max: \033[1;31m%-6.1f\033[0m\n", curr/1024, avg/1024, max/1024
    }' $HIST_FILE_RAM

    echo "-----------------------------------------------------------------------------"

    # --- 3.2 GPU (GB & %) ---
    # Dynamic header: Use variable $AVG_LABEL
    printf "\033[1;34m%-6s | %-30s | %-25s\033[0m\n" "ID" "--- VRAM Usage (GB) ---" "--- GPU Utilization (%) ---"
    printf "\033[1;34m%-6s | %-8s %-10s %-8s | %-7s %-8s %-7s\033[0m\n" "Index" "Curr" "$AVG_LABEL" "Max" "Curr" "Avg" "Max"
    echo "-----------------------------------------------------------------------------"

    # AWK calculations
    awk -F', ' -v gcount="$GPU_COUNT" '{
        id=$1; mem=$2; util=$3;
        
        # VRAM (Raw MB)
        m_sum[id]+=mem; m_count[id]++;
        if(mem > m_max[id]) m_max[id]=mem;
        m_curr[id]=mem;
        
        # Utilization (%)
        u_sum[id]+=util; u_count[id]++;
        if(util > u_max[id]) u_max[id]=util;
        u_curr[id]=util;
        
    } END {
        for(i=0; i < gcount; i++) {
            if(m_count[i] > 0) {
                m_avg = m_sum[i]/m_count[i];
                u_avg = u_sum[i]/u_count[i];
                
                # Output (Divide by 1024 to convert to GB)
                printf "GPU %-2d | %-8.1f %-10.1f %-8.1f | %-7d %-8.1f %-7d\n", 
                    i, m_curr[i]/1024, m_avg/1024, m_max[i]/1024, u_curr[i], u_avg, u_max[i]
            }
        }
    }' $HIST_FILE_GPU

    echo "-----------------------------------------------------------------------------"
    echo "Refresh: ${INT}s | Window: ${WINDOW}s | Samples: ${SAMPLES}"
    
    sleep $INT
done