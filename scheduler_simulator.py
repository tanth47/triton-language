
# A Scheduler Simulator for LLM Serving (simulate vllm V1 scheduler behavior)

######################################################
# Scheduler Simulator
######################################################
def scheduler_simulator(ISL, OSL, concurrency, scale, total_KV_budget, max_num_batched_tokens, custom_scheduler=False):

    full_decode_output = []
    prefill_output = []
    request_waiting_from_client = concurrency * (scale - 1)
    waiting = [ISL] * concurrency
    running = [] # (num_prefilled_token, num_decoded_tokens)
    total_KV_budget_remaining = total_KV_budget

    scheduler_step = 0
    tailing_count = 0
    while waiting or running or request_waiting_from_client > 0:
        # at the first step of online benchmark, only 1 request in waiting queue, so only schedule ISL tokens
        token_budget = max_num_batched_tokens if scheduler_step > 0 else ISL
        scheduler_output = []
        scheduler_step += 1
        is_full_decode = True
        prefill_state = []
        
        # print(f"Step {scheduler_step}: waiting={len(waiting)}, running={len(running)}, request_waiting_from_client={request_waiting_from_client}, total_KV_budget_remaining={total_KV_budget_remaining}, token_budget={token_budget}")

        # Custom scheduler logic: if we have a lot of KV budget remaining, try to schedule more waiting requests first        
        if custom_scheduler and (total_KV_budget_remaining - token_budget)/total_KV_budget > 0.2 and len(waiting) > 0:
            req_index = 0
            while req_index < len(waiting) and token_budget > 0:
                w = waiting[req_index]
                num_new_tokens = min(token_budget, w)

                if total_KV_budget_remaining < num_new_tokens:
                    break

                running.append((num_new_tokens, 0))
                scheduler_output.append(num_new_tokens)

                is_full_decode = False
                req_index += 1
                token_budget -= num_new_tokens
                total_KV_budget_remaining -= num_new_tokens

            waiting = waiting[req_index:]

        # First, try to fill up the running requests
        has_preempted = False
        req_index = 0
        while req_index < len(running) and token_budget > 0:
            r = running[req_index]
            if r[0] == ISL:
                token_needed = 1
            else:
                token_needed = ISL - r[0]
                assert r[1] == 0

            num_new_tokens = min(token_budget, token_needed)

            while True:
                    # simulate allocation of KV memory
                if total_KV_budget_remaining < num_new_tokens:
                    prempempted = running.pop()
                    total_KV_budget_remaining += prempempted[0] + prempempted[1]
                    waiting.append(ISL)
                    has_preempted = True
                        
                    if len(running) == req_index:
                        can_schedule = False
                        break
                else:
                    can_schedule = True
                    break
                
            if not can_schedule:
                break

            if r[0] < ISL:
                assert r[0] + num_new_tokens <= ISL
                if r[0] + num_new_tokens == ISL:
                    running[req_index] = (ISL, 1)
                else:
                    running[req_index] = (r[0] + num_new_tokens, r[1])
                is_full_decode = False
                prefill_state.append((num_new_tokens, r[0]))
            else:
                running[req_index] = (r[0], r[1] + 1)

            token_budget -= num_new_tokens
            total_KV_budget_remaining -= num_new_tokens

            scheduler_output.append(num_new_tokens)

            req_index += 1

        # Next, try to fill up the waiting requests
        if not has_preempted:
            req_index = 0
            while req_index < len(waiting) and token_budget > 0:
                w = waiting[req_index]
                num_new_tokens = min(token_budget, w)

                if total_KV_budget_remaining < num_new_tokens:
                    break

                running.append((num_new_tokens, 0))
                scheduler_output.append(num_new_tokens)

                is_full_decode = False
                prefill_state.append((num_new_tokens, 0))
                req_index += 1
                token_budget -= num_new_tokens
                total_KV_budget_remaining -= num_new_tokens

            waiting = waiting[req_index:]    

        if is_full_decode:
            requests_state = [(1, r[0] + r[1] - 1) for r in running]
            full_decode_output.append(requests_state)
        else:
            if prefill_state:
                prefill_output.append(prefill_state)

        max_scheduled_token = max(scheduler_output) if scheduler_output else 0
        if max_scheduled_token == 1 and len(scheduler_output) < concurrency:
            tailing_count += 1

        # Handle Finished request
        previous = len(running)
        running = [(r[0], r[1]) for r in running if r[0] + r[1] < ISL + OSL]
        finished = previous - len(running)
        total_KV_budget_remaining += finished * (ISL + OSL)

        for _ in range(finished):
            if request_waiting_from_client > 0:
                waiting.append(ISL)
                request_waiting_from_client -= 1

    return full_decode_output, prefill_output

def get_scheduler_output(context_len, batch_size, OSL=512):
    ISL = context_len
    concurrency = batch_size
    scale = 1
    total_KV_budget = 1_836_352 * 8
    max_num_batched_tokens = 16384
    custom_scheduler = False

    return scheduler_simulator(ISL, OSL, concurrency, scale, total_KV_budget, max_num_batched_tokens, custom_scheduler)

def main():
    context_len = 1024
    batch_size = 16
    full_decode_output, prefill_output = get_scheduler_output(context_len, batch_size)
    print("Full Decode Output:")
    for step, state in enumerate(full_decode_output):
        print(f"Step {step+1}: {state}")
    print("\nPrefill Output:")
    for step, state in enumerate(prefill_output):
        print(f"Step {step+1}: {state}")

if __name__ == "__main__":
    main()