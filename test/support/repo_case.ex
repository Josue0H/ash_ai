defmodule AshAi.RepoCase do
  @moduledoc false
  use ExUnit.CaseTemplate

  alias Ecto.Adapters.SQL.Sandbox

  using do
    quote do
      alias AshAi.TestRepo

      import Ecto
      import Ecto.Query
      import AshAi.RepoCase
    end
  end

  setup tags do
    :ok = Sandbox.checkout(AshAi.TestRepo)

    if !tags[:async] do
      Sandbox.mode(AshAi.TestRepo, {:shared, self()})
    end

    :ok
  end
end
