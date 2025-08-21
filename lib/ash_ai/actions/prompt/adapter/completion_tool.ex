defmodule AshAi.Actions.Prompt.Adapter.CompletionTool do
  @moduledoc """
  An adapter that provides a "complete_request" tool that the LLM must call within `max_runs` messages to complete the request.

  ## Adapter Options

  - `:max_runs` - The maximum number of times to allow the LLM to repeatedly generate responses/call tools
    before the action is considered failed.
  """
  @behaviour AshAi.Actions.Prompt.Adapter
  # ignoring dialyzer warning on Message.new_system! as it will be fixed in next release: https://github.com/brainlid/langchain/pull/315
  @dialyzer {:nowarn_function, [run: 2]}

  alias AshAi.Actions.Prompt.Adapter.Data
  alias LangChain.Chains.LLMChain

  def run(%Data{} = data, opts) do
    messages = data.messages

    max_runs = opts[:max_runs] || 25

    {llm, deterministic?} =
      if Map.has_key?(data.llm, :tool_choice) && Enum.empty?(data.tools) do
        {%{data.llm | tool_choice: %{"type" => "tool", "name" => "complete_request"}}, true}
      else
        {data.llm, false}
      end

    description =
      if deterministic? do
        "Use this tool to complete the request."
      else
        """
        Use this tool to complete the request.
        This tool must be called after #{opts[:max_runs] || 25} messages or tool calls from you.
        """
      end

    # Gemini can return MALFORMED_FUNCTION_CALL when schemas are overly strict.
    # For GoogleAI, accept any object for `result` and validate on the server.
    llm_mod_str = to_string(llm.__struct__)
    result_schema_for_provider =
      if String.ends_with?(llm_mod_str, "ChatGoogleAI") do
        %{"type" => "object"}
      else
        data.json_schema
      end

    # Compute required list for parameters
    required_keys_for_provider =
      if String.ends_with?(llm_mod_str, "ChatGoogleAI") do
        []
      else
        ["result"]
      end

    completion_tool =
      LangChain.Function.new!(%{
        name: "complete_request",
        description: description,
        parameters_schema: %{
          "type" => "object",
          "properties" => %{"result" => result_schema_for_provider},
          "required" => required_keys_for_provider,
          "additionalProperties" => false
        },
        strict: true,
        function: fn arguments, _context ->
          with {:ok, value} <-
                 Ash.Type.cast_input(
                   data.input.action.returns,
                   arguments["result"],
                   data.input.action.constraints
                 ),
               {:ok, value} <-
                 Ash.Type.apply_constraints(
                   data.input.action.returns,
                   value,
                   data.input.action.constraints
                 ) do
            {:ok, "complete", value}
          else
            _error ->
              {:error,
               """
               Invalid response. Expected to match schema:

               #{inspect(data.json_schema, pretty: true)}
               """}
          end
        end
      })

    %{
      llm: llm,
      verbose: data.verbose?,
      custom_context: Map.new(Ash.Context.to_opts(data.context))
    }
    |> LLMChain.new!()
    |> AshAi.Actions.Prompt.Adapter.Helpers.add_messages_with_templates(messages, data)
    |> then(fn chain ->
      # Sanitize tools if using Gemini
      tools = [completion_tool | data.tools]
      tools = AshAi.sanitize_tools_for_llm(tools, chain.llm)
      LLMChain.add_tools(chain, tools)
    end)
    |> LLMChain.run_until_tool_used("complete_request", max_runs: max_runs)
    |> case do
      # Normal success shape
      {:ok, _chain, %{processed_content: content}} ->
        {:ok, content}

      # Some langchain versions may return the error wrapped in an :ok tuple with a keyword list
      {:ok, [error: error]} ->
        {:error, error}

      # Pass-through error shape
      {:error, _chain, error} ->
        {:error, error}

      # Fallback: surface an "unexpected_response" error without crashing the process
      other ->
        {:error,
         %LangChain.LangChainError{type: "unexpected_response", message: "Unexpected response", original: other}}
    end
  end
end
