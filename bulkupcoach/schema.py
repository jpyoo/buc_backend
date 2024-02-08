import graphene
import graphapi.schema


class Query(graphapi.schema.Query, graphene.ObjectType):
    # Combine the queries from different apps
    pass


class Mutation(graphapi.schema.Mutation, graphene.ObjectType):
    # Combine the mutations from different apps
    pass


schema = graphene.Schema(query=Query, mutation=Mutation)